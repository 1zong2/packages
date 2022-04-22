import gdown

# ex) job name : https://drive.google.com/uc?id={공유 link}

weight_dic = {
    'face_seg': 'https://drive.google.com/uc?id=166yC9tK7VXRZJGhBk8I66ewl1Fa3L6_3',
    'head_pose': 'https://drive.google.com/uc?id=1AH9cJCPAbov1OxtXPaihqhgveZIGjF7U',
    'swinIR': 'https://drive.google.com/uc?id=1hQSl6iwmlhD6cR5qoMHc7TMqSxRZwHS6',
    'gfpgan': 'https://drive.google.com/uc?id=1fyXxJzQzh00WJbZEvUj_utFXIZt29_4Y',
    'arcface': 'https://drive.google.com/uc?id=1ooTAvChl6j4YA1ohFtE87pKDTeZTsH4K',
    'curricularface': 'https://drive.google.com/uc?id=1WZobwG8vVFf0l-7orUzvYWl6YAWOnx2k'
}

save_path = {
    'face_seg': './packages/face_seg/ptnn/face_seg.pth',
    'head_pose': './packages/head_pose/ptnn/hopenet.pkl',
    'swinIR': './packages/swinIR/ptnn/swinIR.pth',
    'gfpgan': './packages/gfpgan/ptnn/GFPGAN.pth',
    'arcface': './packages/arcface/ptnn/arcface.pth',
    'curricularface': './packages/curricularface/ptnn/curricularface.pth',
}

def download_weight(job):
    gdown.download(weight_dic[job], output=save_path[job], quiet=False)
