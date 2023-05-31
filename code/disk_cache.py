import pickle
from dataset import SceneTextDataset
from east_dataset import EASTDataset
from tqdm import tqdm

train_dataset = SceneTextDataset(
    "/opt/ml/input/data/medical", ## 데이터가 저장되어있는 경로
    split="train", ## ufo json 파일 이름
    image_size=2048,
    crop_size=1024,
    ignore_tags=["masked", "excluded-region", "maintable", "stamp"],
)

train_data = EASTDataset(train_dataset)
for i in tqdm(range(len(train_data))):
    g = train_data.__getitem__(i)
    with open(file=f"/opt/ml/input/data/pkls/{i}.pkl", mode="wb") as f: ## 생성될 pkl 파일을 저장할 경로 -> pkls폴더에 100개의 pkl파일 생성됨
        pickle.dump(g, f)