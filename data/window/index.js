const WEBGPU_AVAILABLE = typeof navigator < 'u' && 'gpu' in navigator;

const proceed = file => {
  proceed.file = file;

  const {canvas, image} = self;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  image.onload = async () => {
    try {
      const {AutoModel, AutoProcessor, env, RawImage} = await import('./transformers/transformers.js');
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = new URL('models', location.href).href;
      env.backends.onnx.wasm.wasmPaths = new URL('transformers', location.href).href + '/';
      env.backends.onnx.wasm.proxy = false;
      env.backends.onnx.wasm.numThreads = 1;

      const oImage = await RawImage.fromBlob(image);
      const {width, height} = oImage;

      canvas.width = width;
      canvas.height = height;

      ctx.font = '5vw Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Please wait...', width / 2, height / 2);

      const model = await AutoModel.from_pretrained('briaai/RMBG-1.4', {
        config: {
          model_type: 'custom'
        },
        quantized: false,
        device: WEBGPU_AVAILABLE ? 'webgpu' : 'wasm'
      });
      const processor = await AutoProcessor.from_pretrained('briaai/RMBG-1.4', {
        config: {
          do_normalize: !0,
          do_pad: !1,
          do_rescale: !0,
          do_resize: !0,
          image_mean: [.5, .5, .5],
          feature_extractor_type: 'ImageFeatureExtractor',
          image_std: [1, 1, 1],
          resample: 2,
          rescale_factor: .00392156862745098,
          size: {
            width: 1024,
            height: 1024
          }
        }
      });

      const {pixel_values: n} = await processor(oImage);
      const {output: d} = await model({
        input: n
      });
      const rImage = await RawImage.fromTensor(d[0].mul(255).to('uint8')).resize(width, height);

      ctx.drawImage(oImage.toCanvas(), 0, 0);

      const p = ctx.getImageData(0, 0, width, height);
      for (let m = 0; m < rImage.data.length; ++m) {
        p.data[4 * m + 3] = rImage.data[m];
      }
      ctx.putImageData(p, 0, 0);

      self.download.disabled = false;
    }
    catch (e) {
      console.error(e);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillText('Error!', canvas.width / 2, canvas.height / 2);
      alert(e.message);
    }
  };

  self.download.disabled = true;
  image.src = typeof file === 'string' ? file : URL.createObjectURL(file);
};

self.file.onchange = e => {
  proceed(e.target.files[0]);
};

document.ondragover = e => {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
};
document.ondrop = e => {
  e.preventDefault();
  const files = e.dataTransfer.files;
  proceed(files[0]);
};

self.download.onclick = () => {
  const link = document.createElement('a');
  link.download = proceed.file.name;
  link.href = self.canvas.toDataURL('image/png');
  link.click();
};

self.sample.onclick = () => proceed('./samples/input.jpg');
