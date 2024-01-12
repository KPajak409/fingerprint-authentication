#%%
import others.take_scan as scan
from pathlib import Path


# at first read and save user fingerprint scan and as success return path to saved image
# it obviously won't work if scanner isn't connected, you can use path to existing file instead in order to test scan_preprocessing
path = scan.getPrint()

# later run preprocess on user scan by sending his id as argument
scan.scan_preprocess(path)

# finally you can load preprocessed user fingerprint scan by loading it from the path project_dir/scans/processed/user_id.bmp