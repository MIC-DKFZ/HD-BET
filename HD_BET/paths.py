import os

# please refer to the readme on where to get the parameters. Save them in this folder:
folder_with_parameter_files = os.path.join(os.path.expanduser('~'), 'hd-bet_params', 'release_2.0.0')

# Pretrained weights are hosted on Zenodo: https://zenodo.org/records/14445620 (DOI 10.5281/zenodo.14445620),
# licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
# the link suggests 1.5but I don't want to release this update as v1.5 because this will upset people who use hd-bet
# and who don't expect breaking changes without a major version upgrade
ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1'
