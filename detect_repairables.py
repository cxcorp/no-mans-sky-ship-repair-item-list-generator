import sys
import os
import cv2 as cv
import numpy as np
import pandas as pd

from trackbars import ChangeTrackingTrackbars
import ColorDescriptor
import Searcher

IMG_A = "./screenies/a_1440p.png"

EXIT_KEY = ord('q')


def rgb2lab(rgb):
    lab = cv.cvtColor(np.uint8([[rgb]]), cv.COLOR_RGB2Lab)
    return lab[0][0]


def getRedMaskExpansion(imgHeight):
    # regression line fit for
    #
    # | Screen | Inventory  |
    # | height | tile width |
    # | ------ | ---------- |
    # |    768 |         66 |
    # |   1080 |         93 |
    # |   1440 |        126 |
    #
    return int(0.09 * imgHeight - 3)


def readTemplateFile(name):
    img = cv.imread(f'./images/{name}')
    return cv.resize(img, (60, 60), interpolation=cv.INTER_LANCZOS4)


def doThing():
    img = cv.imread(IMG_A)

    templates = [f for f in os.listdir(
        'images') if os.path.isfile(os.path.join('images', f))]
    templateImages = [
        (os.path.splitext(template)[0], readTemplateFile(template)) for template in templates]

    if img is None:
        sys.exit("Couldn't read image")

    cv.namedWindow('image')

    tbs = ChangeTrackingTrackbars('image')

    ## Automatically find our area of interest (rectangle within screenshot which includes the inventory tiles)
    # If doesn't work, could replace with semiautomatic "drag to select inventory from screenshot".
    #
    # Find the area of interest by finding the specific red used in the broken item backgrounds
    # + nearby colors as compared in Lab color space (better for visual similarity; better results than
    # RGB comparison for compression schemes like what JPEG uses)
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    lab = cv.cvtColor(blurred, cv.COLOR_BGR2Lab)
    redSample = (123, 0, 0)
    redSampleLab = rgb2lab(redSample)
    redTreshold = 4
    lower = np.clip([x - redTreshold for x in redSampleLab], 0, 255)
    upper = np.clip([x + redTreshold for x in redSampleLab], 0, 255)
    mask = cv.inRange(lab, lower, upper)

    # Now `mask` is just a mask of the areas with the red bg color.
    # Dilate the mask so that if the farthest corner of the item is the only
    # place where the red color exists, the mask will cover the entire theoretical
    # distance to the edge of the inventory slot.
    maskThreshold = getRedMaskExpansion(img.shape[0])
    mask = cv.dilate(mask, cv.getStructuringElement(
        cv.MORPH_RECT, (maskThreshold, maskThreshold)))

    # Then calculate a bounding rectangle which covers the entire
    # dilated mask so that we get a nice rectangle image instead of
    # a broken one with parts missing out. Use this as the mask.
    (x, y, w, h) = cv.boundingRect(mask)
    mask[:] = (0,)
    mask[y:y+h, x:x+w] = (255,)

    tbs.changed = True

    tbs.add("rgb").min(0).max(2).initial(0).build()
    # tbs.add("d").min(1).max(50).initial(9).build()
    # tbs.add("sigmaColor").min(1).max(150).initial(75).build()
    # tbs.add("sigmaSpace").min(1).max(150).initial(75).build()
    tbs.add('alpha').min(0).max(300).initial(300).build()
    tbs.add('beta').min(-100).max(100).initial(100).build()
    tbs.addUint8('treshold').initial(50).build()
    tbs.add('kernel').min(1).max(49).initial(3).build()
    tbs.add('close').min(1).max(49).initial(5).build()

    imgLowRez = cv.resize(img, (1280, 720), interpolation=cv.INTER_LANCZOS4)
    maskLowRez = cv.resize(mask, (1280, 720), interpolation=cv.INTER_LINEAR)

    x, y, w, h = cv.boundingRect(maskLowRez)

    while True:
        if tbs.changed:
            # print('changed')
            tbs.changed = False

            cropped = imgLowRez[y:y+h, x:x+w]
            result = cropped

            # Add contrast and darken the image so that the bilateral filter
            # is more effective
            # a = 2.52
            # b = -30
            alpha = tbs.getPos('alpha') / 100.0
            beta = tbs.getPos('beta')
            result = cv.convertScaleAbs(result, alpha=alpha, beta=beta)
            
            # Blur image
            # values found experimentally via a trackbar - these values
            # seem to provide the clearest edges with the laplacian.
            result = cv.bilateralFilter(
                result, d=23, sigmaColor=83, sigmaSpace=89)

            result = cv.split(result)[tbs.getPos('rgb')]

            kernel = tbs.getPos('kernel')
            if kernel < 1:
                kernel = 1
            if kernel % 2 == 0:
                kernel += 1
            
            # Run edge detection with Laplacian + normal threshold after the fact
            result = cv.Laplacian(result, cv.CV_64F, ksize=kernel)
            result = cv.convertScaleAbs(np.absolute(result))
            _, result = cv.threshold(result,
                                     tbs.getPos('treshold'),
                                     255,
                                     cv.THRESH_BINARY)

            close = tbs.getPos('close')
            if close % 2 == 0:
                close += 1

            # Close (dilate, then erode) the thresholded image to remove
            # artifacts from Laplacian and other variations from the blur
            # -> small nearby pixels get merged
            result = cv.morphologyEx(result, cv.MORPH_CLOSE, cv.getStructuringElement(
                cv.MORPH_RECT, (close, close)))

            ## Find the rectangles.
            # Fill up gaps. Can't just take EXTERNAL contours because there's
            # rubbish because some corners aren't filled.
            contours, _ = cv.findContours(
                result, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                # fill all contour bounding rects as white -> merge contours in practice
                cv.rectangle(result, (x, y), (x+w, y+h), 255, -1)

            # now get actual rectangle contours
            contours, _ = cv.findContours(
                result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            bboxes = [cv.boundingRect(cnt) for cnt in contours]

            def arc(rect):
                # calculate aspect ratio
                x, y, w, h = rect
                return w/float(h)
            # filter rectangles
            # (take only those whose aspect ratio is within 0.15 of our target "portrait" rectangle)
            bboxes = [b for b in bboxes if abs(0.85 - arc(b)) <= 0.15]

            result = cropped.copy()

            tiles = []
            i = 0
            for x, y, w, h in bboxes:
                tile = cropped[y:y+h, x:x+w]
                # grab middle bottom 60x60px square to extract
                # only the icon so that the scrolling text on the top
                # doesn't mess with our feature detection
                h, w, _ = tile.shape
                ystart = max(0, h - 61) if h > 60 else 0
                yend = ystart + 60
                xstart = (w - 60) // 2 if w > 60 else 0
                xend = xstart+60
                tile = tile[ystart:yend, xstart:xend]
                cv.imwrite(f'tiles/{i:02}.png', tile)
                tiles.append((tile, (x, y, w, h)))
                i += 1

            # 8 bins for H, 12 bins for S, 13 bins for V in HSV color
            cd = ColorDescriptor.ColorDescriptor((8, 12, 13))

            featureIndex = []
            for templateName, template in templateImages:
                # Calculate features for our template (target) images
                # This should be done outside doThing() but it's so fast that whatever.
                # The features we use are just based on HSV histograms and they're good
                # enough for us because we preprocess the image so well that we could
                # *almost* compare plainly by pixel values. Almost.
                features = cd.describe(template)
                featureIndex.append((templateName, np.array(features)))

            s1 = Searcher.Searcher("")
            detected_tiles = {}
            for i, (tile, bbox) in enumerate(tiles):
                features = np.array(cd.describe(tile))
                results = s1.searchIdx(featureIndex, features)

                score, name = results[0]
                # 0.7 determined experimentally to be a good threshold for "not actual match"
                color = (0, 255, 0) if score < 0.7 else (0, 0, 255)
                # draw debug rectangles & results
                cv.rectangle(
                    result, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
                cv.putText(
                    result, f'{name}', (bbox[0], bbox[1]+15), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 5)
                cv.putText(
                    result, f'{score:.2f}', (bbox[0], bbox[1]+15+15), cv.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 5)
                cv.putText(
                    result, f'{name}', (bbox[0], bbox[1]+15), cv.FONT_HERSHEY_PLAIN, 0.5, color, 1)
                cv.putText(
                    result, f'{score:.2f}', (bbox[0], bbox[1]+15+15), cv.FONT_HERSHEY_PLAIN, 0.7, color, 1)
                if score < 0.7:
                    # add +1 to this detected item
                    detected_tiles[name] = detected_tiles.get(name, 0) + 1

            df = pd.read_csv('materials.csv', sep=',')
            materials = df.groupby('Name')

            print(detected_tiles)

            objs = []
            for name, amount in detected_tiles.items():
                group = materials.get_group(name)
                group = group.copy()
                group["Amount"] *= amount
                objs.append(group[['Material', 'Amount']])
            print(pd.concat(objs).groupby('Material').agg('sum'))

            cv.imshow('image', result)
        k = cv.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            break

    cv.destroyAllWindows()
    sys.exit("exiting")


doThing()
