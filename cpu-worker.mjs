import { parentPort } from 'node:worker_threads';

if (!parentPort) {
  throw new Error('cpu-worker.mjs must be started as a Worker thread.');
}

parentPort.postMessage({ type: 'ready' });

parentPort.on('message', (message) => {
  if (message?.type !== 'run') return;

  const {
    runId,
    inputBuffer,
    outputBuffer,
    controlBuffer,
    width,
    height,
    radius,
    sigmaSpaceSquared,
    sigmaColorSquared,
    chunkRows,
  } = message;

  try {
    const input = new Uint8Array(inputBuffer);
    const output = new Uint8Array(outputBuffer);
    const nextRow = new Int32Array(controlBuffer);

    /*
     * Dynamic row-chunk scheduling:
     * Atomics.add assigns disjoint row ranges to workers. This keeps the
     * implementation race-free while making chunkRows a tuning parameter.
     */
    while (true) {
      const rowStart = Atomics.add(nextRow, 0, chunkRows);
      if (rowStart >= height) break;
      const rowEnd = Math.min(height, rowStart + chunkRows);

      for (let y = rowStart; y < rowEnd; ++y) {
        for (let x = 0; x < width; ++x) {
          const centerIndex = (y * width + x) * 4;
          const centerR = input[centerIndex + 0];
          const centerG = input[centerIndex + 1];
          const centerB = input[centerIndex + 2];

          let weightedR = 0;
          let weightedG = 0;
          let weightedB = 0;
          let totalWeight = 0;

          for (let dy = -radius; dy <= radius; ++dy) {
            const sampleY = Math.max(0, Math.min(height - 1, y + dy));

            for (let dx = -radius; dx <= radius; ++dx) {
              const sampleX = Math.max(0, Math.min(width - 1, x + dx));
              const sampleIndex = (sampleY * width + sampleX) * 4;

              const sampleR = input[sampleIndex + 0];
              const sampleG = input[sampleIndex + 1];
              const sampleB = input[sampleIndex + 2];

              const spatialDistanceSquared = dx * dx + dy * dy;
              const dr = sampleR - centerR;
              const dg = sampleG - centerG;
              const db = sampleB - centerB;
              const colorDistanceSquared = dr * dr + dg * dg + db * db;

              const exponent =
                -(spatialDistanceSquared / sigmaSpaceSquared)
                -(colorDistanceSquared / sigmaColorSquared);
              const weight = Math.exp(exponent);

              weightedR += sampleR * weight;
              weightedG += sampleG * weight;
              weightedB += sampleB * weight;
              totalWeight += weight;
            }
          }

          output[centerIndex + 0] = Math.trunc(Math.max(0, Math.min(255, weightedR / totalWeight)));
          output[centerIndex + 1] = Math.trunc(Math.max(0, Math.min(255, weightedG / totalWeight)));
          output[centerIndex + 2] = Math.trunc(Math.max(0, Math.min(255, weightedB / totalWeight)));
          output[centerIndex + 3] = 255;
        }
      }
    }

    parentPort.postMessage({ type: 'done', runId });
  } catch (error) {
    parentPort.postMessage({
      type: 'error',
      runId,
      error: error?.stack || String(error),
    });
  }
});
