from PIL import Image, ImageDraw, ImageFont
from . import utils


class BboxEllipse:
    def __init__(self, args):
        self.pickle_file = utils.read_file(args.pickle_path)
        self.threshold = args.threshold
        self.output_path = args.output_path
        self.image_path = args.image_path
        self.scale = args.ellipse_scale
        self.json_data = utils.read_file(args.json_path)

    def __create_mapping_paths(self, json_data, image_path):
        mapping = {}
        image_paths = []
        for i, image in enumerate(json_data['images']):
            mapping[image['file_name'].split('/')[-1]] = i
            image_paths.append(f"{image_path}/{image['file_name']}")
        return mapping, image_paths

    def __draw_point(self, img, bboxes):
        draw = ImageDraw.Draw(img, mode='RGB')
        for bbox in bboxes:
            coords = utils.get_coordinates(bbox, self.scale)

            if bbox[4] > self.threshold:
                draw.ellipse((coords.x0, coords.y0, coords.x1, coords.y1), fill=utils.acc_to_color(bbox[4]))
        return img

    def __draw_text(self, img, bboxes):
        draw = ImageDraw.Draw(img, mode='RGB')
        for bbox in bboxes:
            coords = utils.get_coordinates(bbox, self.scale)
            fnt = ImageFont.truetype("/usr/share/fonts/liberations/LiberationSans-Regular.ttf", 16)  # 16
            draw.text((coords.xm, coords.ym), bbox.pred_class[:5], align='left', font=fnt, fill=utils.acc_to_color(bbox.pred_score), stroke_width=1, stroke_fill=utils.acc_to_color(bbox.det_pred))
        return img

    def convert_using_pkl(self):
        new_images = []
        mapping, image_paths = self.__create_mapping_paths(self.json_data, self.image_path)

        for image in image_paths:
            img = Image.open(image)
            image_name = image.split('/')[-1]

            # Draw Points
            img = self.__draw_point(img, self.pickle_file[mapping[image_name]][0])
            new_images.append(img)
        return new_images, image_paths

    def convert_using_json(self):
        new_images = []
        image_paths = []
        for det_image in self.json_data:
            try:
                image_path = det_image[0]['detection_image']
            except Exception:
                continue
                print(image_path)
            i_path = self.image_path+'/'+'/'.join(image_path)
            image_paths.append(i_path)
            img = Image.open(i_path)

            bboxes = utils.collect_bboxes(det_image)

            img = self.__draw_text(img, bboxes)
            new_images.append(img)
        return new_images, image_paths

    def convert(self):
        if self.pickle_file is not None:
            return self.convert_using_pkl()
        elif self.json_data is not None:
            return self.convert_using_json()


if __name__ == '__main__':
    argparser = utils.create_argparse()

    bbox_ellipse = BboxEllipse(argparser)
    images, paths = bbox_ellipse.convert()
    utils.save_images(images, paths, argparser.output_path)
