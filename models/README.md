# Trained checkpoint placement

Put your trusted trained files here, maintaining the original filenames:

```
models/disease/maxvit_tiny_tf_224_best.pth
models/disease/swin_tiny_patch4_window7_224_best.pth
models/disease/convnextv2_tiny_best.pth
models/disease/mambaout_tiny_best.pth
models/nutrient/maxvit_tiny_recovered.pth
models/nutrient/swin_best.pth
models/nutrient/convnextv2_tiny_final_checkpoint.pth
models/nutrient/mambaout_tiny_final_checkpoint.pth
```

The same filenames in a `checkpoints` subdirectory are supported. Supply verified per-checkpoint `.classes.json` sidecars or a `classes.json` shared only by models trained with that same class order. No checkpoint files are included in this repository.

On Windows, the folders `D:\rocole_github_deploy\models\nutrient` and `D:\rocole_github_deploy\models\disease` are automatically used if present and the checkout has no corresponding models folder. Optional process variables `ROCOLE_NUTRIENT_DIR` and `ROCOLE_DISEASE_DIR` override discovery and can point to mounted model directories.

On Streamlit Cloud or another Linux server, a Windows D: path cannot be accessed. Upload or securely download the files onto that server. A Git push of application code alone does not transfer your local model folders. For files exceeding your hosting provider's repository limits, use an authenticated model artifact store or a supported mounted volume. The app does not download arbitrary checkpoints or bypass strict checkpoint validation.

The app now automatically attempts a Drive download for missing models when running a scan. No separate download button is shown. For server setup, `python checkpoint_download.py` attempts all eight. Successful acquisition still requires accessible Drive sharing and exact expected filenames.
