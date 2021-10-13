import coco_text
import skimage.io as io
import math
import config
import html


dataDir = config.DATA_DIR
dataType = 'train2014'

ct = coco_text.COCO_Text(config.JSON_DIR)
ct.info()
imgIds = ct.getImgIds(imgIds=ct.val, catIds=[('legibility', 'legible')])

f = open("./coco-text_dataset/val.txt", "w", encoding="utf-8")

print("!!!", len(imgIds))

count = 0
for idx in range(0, len(imgIds)):
    img = ct.loadImgs(imgIds[idx])[0]
    I = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))

    if idx % 100 == 0:
        print(idx, "/", len(imgIds))

    annIds = ct.getAnnIds(imgIds=img['id'])
    anns = ct.loadAnns(annIds)

    for gt in anns:
        if "utf8_string" not in gt or gt["language"] != "english":
            continue
        label = gt["utf8_string"]
        label = label.strip()
        if label == "" or label == " ":
            continue
        label = html.unescape(label)
        bb = gt["bbox"]
        x = math.floor(bb[0])
        y = math.floor(bb[1])
        dx = math.ceil(bb[2])
        dy = math.ceil(bb[3])
        xx = x + dx
        yy = y + dy
        if xx >= I.shape[1]:
            xx = I.shape[1] - 1
        if yy >= I.shape[0]:
            yy = I.shape[0] - 1
        if len(I.shape) == 2:
            croppedimg = I[y:yy, x:xx]
        else:
            croppedimg = I[y:yy, x:xx, :]
        count += 1
        url = 'D:/Visual_Text_Recognition_using_COCO_text/coco-text_dataset/val/' + str(count) + ".jpg"
        gt_url = "./val/" + str(count) + ".jpg"
        try:
            io.imsave(url, croppedimg)
            f.write(url +" " + label + "\n")
        except Exception as ex:
            count -= 1

            print(ex)

f.close()
