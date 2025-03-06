### STEP 1 切patch

采用两步法确保每张切片都被成功分割前景。

首先使用如下命令仅执行分割：

```bash
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --step_size 256 --seg --preset bwh_biopsy.csv
```

此命令会产生一个`process_list_autogen.csv`，保存每张切片的分割参数和状态。

诸张切片检查分割结果，对于分割不满意的切片，修改`process_list_autogen.csv`中的参数，并将`process`列改为1，然后执行如下命令重新分割并检查：

```bash
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --step_size 256 --seg --process_list process_list_autogen.csv
```

检查无误后，将所有`process`列改为1，然后执行如下命令分割并切patch：

```bash
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --step_size 256 --seg --process_list process_list_autogen.csv --patch --stitch
```

### STEP 2 过滤patch

为了删除组织含量很少的patch，执行如下命令：

```bash
python filter_patches_fp.py --data_h5_dir DATA_H5_DIR --data_slide_dir DATA_DIRECTORY --slide_ext .svs --csv_path CSV_FILE_PATH --save_dir RESULTS_DIRECTORY --tissue_ratio 0.4
```

过滤后的patch坐标会被保存在`RESULTS_DIRECTORY`下的`filtered_patches`文件夹中。

### STEP 3 提取特征

执行如下命令：

```bash
python extract_features_stainnorm_fp.py --data_h5_dir DATA_H5_DIR --data_slide_dir DATA_DIRECTORY --norm_target_dir NORM_TARGET_DIR --slide_ext .svs --csv_path CSV_FILE_PATH --feat_dir FEAT_DIRECTORY --model_name uni_v1
```

如果不希望进行染色归一化，可以将`extract_features_stainnorm_fp`改为`extract_features_fp`。