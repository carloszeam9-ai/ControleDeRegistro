
"use strict";

// YOLO Face + OpenCV.js + ONNX Runtime Web
// Totem Escolar - Detecção facial

const YoloFace = (() => {
  let session = null;
  let inputName = null;

  let width = 640;
  let height = 640;

  const isReady = () => session !== null;

  // =====================================
  // CARREGAR MODELO ONNX
  // =====================================

  async function load(file) {
    if (!window.ort) {
      throw new Error(
        "ONNX Runtime Web não foi carregado."
      );
    }

    if (
      !window.cv ||
      typeof cv.Mat !== "function"
    ) {
      throw new Error(
        "OpenCV.js não foi carregado."
      );
    }

    if (!file) {
      throw new Error(
        "Selecione um arquivo .onnx."
      );
    }

    session = null;

    const bytes = new Uint8Array(
      await file.arrayBuffer()
    );

    const s = await ort.InferenceSession.create(
      bytes,
      {
        executionProviders: ["wasm"]
      }
    );

    if (s.inputNames.length !== 1) {
      throw new Error(
        "O modelo deve possuir uma entrada."
      );
    }

    const name = s.inputNames[0];

    
// Obter os metadados da entrada ONNX
const metadata = s.inputMetadata;

console.log("Metadados completos:", metadata);

let meta = null;

// Compatibilidade com diferentes versões
if (metadata instanceof Map) {
  meta = metadata.get(name);
} else if (Array.isArray(metadata)) {
  meta = metadata[0];
} else if (metadata) {
  meta = metadata[name];
}

console.log("Metadados da entrada:", meta);

let dims =
  meta?.dimensions ??
  meta?.dims ??
  meta?.shape ??
  null;

// Algumas versões expõem os metadados
// diretamente em inputMetadata.
if (!dims && metadata) {
  dims =
    metadata.dimensions ??
    metadata.dims ??
    metadata.shape ??
    null;
}

console.log("Dimensões encontradas:", dims);

// Se os metadados não estiverem disponíveis,
// utilizar 640x640 como configuração provisória.
if (!dims) {
  console.warn(
    "Dimensões indisponíveis. " +
    "Tentando entrada [1,3,640,640]."
  );

  dims = [1, 3, 640, 640];
}

    console.log("Entradas ONNX:", s.inputNames);
    console.log("Metadados:", s.inputMetadata);
    console.log("Saídas ONNX:", s.outputNames);

    if (!dims || dims.length !== 4) {
      throw new Error(
        "Formato de entrada inválido: " +
        JSON.stringify(dims)
      );
    }

    // Esperado: [1, 3, H, W]
    // H e W podem ser dinâmicos.

    if (
      dims[0] !== 1 &&
      !isDynamic(dims[0])
    ) {
      throw new Error(
        "Batch incompatível: " +
        JSON.stringify(dims)
      );
    }

    if (dims[1] !== 3) {
      throw new Error(
        "O modelo deve usar 3 canais RGB."
      );
    }

    height = chooseDimension(dims[2]);
    width = chooseDimension(dims[3]);

    if (
      height > 1280 ||
      width > 1280 ||
      height <= 0 ||
      width <= 0
    ) {
      throw new Error(
        "Dimensões não suportadas."
      );
    }

    session = s;
    inputName = name;

    console.log(
      "Modelo carregado:",
      width,
      "x",
      height
    );

    return (
      "Modelo carregado! Entrada " +
      name +
      ": [1,3," +
      height +
      "," +
      width +
      "] | Saídas: " +
      s.outputNames.join(", ")
    );
  }

  function isDynamic(value) {
    return (
      value === null ||
      value === undefined ||
      typeof value === "string" ||
      value === -1
    );
  }

  function chooseDimension(value) {
    if (Number.isInteger(value) && value > 0) {
      return value;
    }

    if (isDynamic(value)) {
      return 640;
    }

    throw new Error(
      "Dimensão ONNX inválida: " + value
    );
  }

  // =====================================
  // PRÉ-PROCESSAMENTO COM OPENCV.JS
  // =====================================

  function preprocess(canvas) {
    let source;
    let resized;
    let padded;

    try {
      source = cv.imread(canvas);

      const sw = source.cols;
      const sh = source.rows;

      if (!sw || !sh) {
        throw new Error(
          "Imagem vazia ou inválida."
        );
      }

      const scale = Math.min(
        width / sw,
        height / sh
      );

      const nw = Math.max(
        1,
        Math.round(sw * scale)
      );

      const nh = Math.max(
        1,
        Math.round(sh * scale)
      );

      const left = Math.floor(
        (width - nw) / 2
      );

      const top = Math.floor(
        (height - nh) / 2
      );

      resized = new cv.Mat();
      padded = new cv.Mat();

      cv.resize(
        source,
        resized,
        new cv.Size(nw, nh),
        0,
        0,
        cv.INTER_LINEAR
      );

      cv.copyMakeBorder(
        resized,
        padded,
        top,
        height - nh - top,
        left,
        width - nw - left,
        cv.BORDER_CONSTANT,
        new cv.Scalar(114, 114, 114, 255)
      );

      const pixels = padded.data;
      const area = width * height;

      const floats = new Float32Array(
        area * 3
      );

      // RGBA -> RGB, formato NCHW

      for (let i = 0; i < area; i++) {
        floats[i] =
          pixels[i * 4] / 255;

        floats[area + i] =
          pixels[i * 4 + 1] / 255;

        floats[2 * area + i] =
          pixels[i * 4 + 2] / 255;
      }

      const tensor = new ort.Tensor(
        "float32",
        floats,
        [1, 3, height, width]
      );

      return {
        tensor,
        scale,
        left,
        top,
        sw,
        sh
      };

    } finally {
      source?.delete();
      resized?.delete();
      padded?.delete();
    }
  }

  // =====================================
  // FUNÇÕES AUXILIARES
  // =====================================

  function clamp(value, min, max) {
    return Math.max(
      min,
      Math.min(max, value)
    );
  }

  function iou(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);

    const x2 = Math.min(
      a.x + a.w,
      b.x + b.w
    );

    const y2 = Math.min(
      a.y + a.h,
      b.y + b.h
    );

    const intersection =
      Math.max(0, x2 - x1) *
      Math.max(0, y2 - y1);

    const union =
      a.w * a.h +
      b.w * b.h -
      intersection;

    return intersection / (union + 1e-9);
  }

  // =====================================
  // DECODIFICAR DETECÇÕES
  // =====================================

  function decode(output, info, threshold = 0.45) {
    const d = output.dims;

    console.log("Formato da saída YOLO:", d);

    if (
      d.length !== 3 ||
      d[0] !== 1
    ) {
      throw new Error(
        "Saída incompatível: " +
        JSON.stringify(d)
      );
    }

    // YOLOv8 de uma classe:
    // [1, 5, N] ou [1, N, 5]

    const channelsFirst = d[1] === 5;
    const channelsLast = d[2] === 5;

    if (!channelsFirst && !channelsLast) {
      throw new Error(
        "Formato YOLO não suportado: " +
        JSON.stringify(d) +
        ". Esperado [1,5,N] ou [1,N,5]."
      );
    }

    const n = channelsFirst ? d[2] : d[1];
    const data = output.data;

    if (n > 200000) {
      throw new Error(
        "Saída do modelo muito grande."
      );
    }

    const get = (i, j) => {
      if (channelsFirst) {
        return data[j * n + i];
      }

      return data[i * 5 + j];
    };

    const boxes = [];

    for (let i = 0; i < n; i++) {
      const score = get(i, 4);

      if (
        !Number.isFinite(score) ||
        score < threshold ||
        score > 1
      ) {
        continue;
      }

      const cx = get(i, 0);
      const cy = get(i, 1);
      const w = get(i, 2);
      const h = get(i, 3);

      if (
        ![cx, cy, w, h].every(Number.isFinite) ||
        w <= 0 ||
        h <= 0
      ) {
        continue;
      }

      const left = clamp(
        (cx - w / 2 - info.left) / info.scale,
        0,
        info.sw
      );

      const top = clamp(
        (cy - h / 2 - info.top) / info.scale,
        0,
        info.sh
      );

      const right = clamp(
        (cx + w / 2 - info.left) / info.scale,
        0,
        info.sw
      );

      const bottom = clamp(
        (cy + h / 2 - info.top) / info.scale,
        0,
        info.sh
      );

      if (
        right > left &&
        bottom > top
      ) {
        boxes.push({
          x: left,
          y: top,
          w: right - left,
          h: bottom - top,
          score
        });
      }
    }

    // Non-Maximum Suppression (NMS)

    boxes.sort(
      (a, b) => b.score - a.score
    );

    const selected = [];

    for (const box of boxes) {
      if (
        selected.every(
          previous => iou(previous, box) < 0.45
        )
      ) {
        selected.push(box);
      }

      if (selected.length >= 30) {
        break;
      }
    }

    return selected;
  }

  // =====================================
  // EXECUTAR DETECÇÃO FACIAL
  // =====================================

  async function detect(canvas) {
    if (!session) {
      throw new Error(
        "Carregue o modelo YOLO primeiro."
      );
    }

    const info = preprocess(canvas);

    const outputs = await session.run({
      [inputName]: info.tensor
    });

    console.log(
      "Saídas do modelo:",
      session.outputNames
    );

    if (session.outputNames.length !== 1) {
      throw new Error(
        "Este código exige uma saída ONNX."
      );
    }

    const outputName = session.outputNames[0];
    const output = outputs[outputName];

    if (!output) {
      throw new Error(
        "Saída ONNX não encontrada."
      );
    }

    const boxes = decode(output, info);

    return {
      boxes,
      shape: output.dims
    };
  }

  return {
    load,
    detect,
    isReady
  };
})();
