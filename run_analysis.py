import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from src.visualisations.visualise_inference import visualise_inference as visualise
from src.visualisations.visualise_inference import tools
import configparser

config = configparser.ConfigParser()
config.read('./configs/setup_kasperpc.config')

cam = config['analysis']['cam']

work_dir = Path(f'./work_dir/{config["analysis"]["work_dir"]}')
visualise.inference(config, 'cam005', work_dir)