import pandas as pd
pd.options.plotting.backend = "plotly"
path_txt = '/home/sa001/workspace/pytorch-CycleGAN-and-pix2pix/checkpoints/sentinel2_pix2pix_256_unet/loss_log.txt'
if __name__ == '__main__':
    file1 = open(path_txt, 'r') 
    lines = file1.readlines()
    dicts = list()
    for i, line in enumerate(lines):
        if i < 1:
            continue
        parts = line.split(') ')[1].split(' ')
        parts.pop(-1)
        dict_tmp = dict()
        dict_tmp['G_GAN'] = float(parts[1])
        dict_tmp['G_L1'] = float(parts[3])
        dict_tmp['D_real'] = float(parts[5])
        dict_tmp['D_fake'] = float(parts[7])
        dicts.append(dict_tmp)
    df = pd.DataFrame(dicts)
    print(df)
    fig = df.plot(title="Sentinel2 pix2pix unet256", labels=dict(index="Steps", value="Losses", variable="Losses"))
    fig.show()


