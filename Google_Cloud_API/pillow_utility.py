from PIL import Image, ImageDraw, ImageFont


def draw_borders(pillow_image, image_size, bounding, color, caption='', confidence_score=0):
    """This method draws a rectangular border around the identified obect"""

    width, height = image_size
    draw = ImageDraw.Draw(pillow_image)
    draw.polygon([
        bounding.normalized_vertices[0].x * width , bounding.normalized_vertices[0].y * height ,
        bounding.normalized_vertices[1].x * width , bounding.normalized_vertices[1].y * height ,
        bounding.normalized_vertices[2].x * width , bounding.normalized_vertices[2].y * height ,
        bounding.normalized_vertices[3].x * width , bounding.normalized_vertices[3].y * height 
    ], fill=None , outline=color)

    font_size = width * height // 22000 if width * height > 40000 else 12

    font = ImageFont.truetype(r'C:/Users/Mota-PC/AppData/Local/Microsoft/Windows/Fonts/Segoe UI.ttf' ,12)

    draw.text((bounding.normalized_vertices[0].x * width, bounding.normalized_vertices[0].y * height + 20), font=font, font_size=font_size, color=color, text=caption)

    draw.text((bounding.normalized_vertices[0].x * width, bounding.normalized_vertices[0].y * height), font=font, font_size=font_size, color=color, 
                text='Confidence Interval :{0:.2f}%'.format(confidence_score))

