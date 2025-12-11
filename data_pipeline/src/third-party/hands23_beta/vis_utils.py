import numpy as np
import pdb, os, cv2
from PIL import Image, ImageDraw, ImageFont


color_rgb  = [(255,255,0), (255, 128,0), (128,255,0), (0,128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127), (255,0,0), (255,204,153), (255,102,102), (153,255,153), (153,153,255), (0,0,153)]
color_rgba = [(255,255,0,70), (255, 128,0,70), (128,255,0,70), (0,128,255,70), (0,0,255,70), (127,0,255,70), (255,0,255,70), (255,0,127,70), (255,0,0,70), (255,204,153,70), (255,102,102,70), (153,255,153,70), (153,153,255,70), (0,0,153,70)]


hand_rgb = [(0, 90, 181), (220, 50, 32)] 
# hand_rgba = [(0, 90, 181, 70), (220, 50, 32, 70)]
hand_rgba = [(0, 90, 181, 128), (220, 50, 32, 160)]

firstobj_rgb = (255, 194, 10)
# firstobj_rgba = (255, 194, 10, 70)
firstobj_rgba = (255, 194, 10, 160)


secondobj_rgb = (0, 159, 115)
# secondobj_rgba = (0, 159, 115, 70)
secondobj_rgba = (0, 159, 115, 160)

txt_color = (0, 0, 0)




def parseSide(s):
    if   s == 'left_hand':  return 'L', 0
    elif s == 'right_hand': return 'R', 1
    else:
        print(f'Weird hand side label is {s}, {type(s)}')
        pdb.set_trace()

def parseState(s):
    if   s == 'no_contact':              return 'NoC',      0
    elif s == 'other_person_contact':    return 'OtherPC', 1
    elif s == 'self_contact':            return 'SelfC',   2
    elif s == 'object_contact':          return 'ObjC',    3
    elif s == 'obj_to_obj_contact':      return 'ObjC',     4
    elif s in ['inconclusive', 'None']:  return  None,     -1
    else:
        print(f'Weird hand state label is {s}')
        pdb.set_trace()

def parseGraspType(s):
    '''Parse String to Int.
    '''
    if s == 'NP-Palm':          return s, 0
    elif s == 'NP-Fin':           return s, 1
    elif s == 'Pow-Pris':         return s, 2
    elif s == 'Pre-Pris':         return s, 3
    elif s == 'Pow-Circ':         return s, 4
    elif s == 'Pre-Circ':         return s, 5
    elif s == 'Later':            return s, 6
    elif s in ['other', 'Other']: return 'Other', 7
    elif s == 'None':             return s, -1
    else:
        print(f'Weird grasp type label is {s}')
        pdb.set_trace()

def parseTouchType(s):
    if s == 'tool_,_touched':        return 'Tool:touched', 0
    elif s == 'tool_,_held':         return 'Tool:held', 1
    elif s == 'tool_,_used':         return 'Tool:used', 2
    elif s == 'container_,_touched': return 'Cont:touched', 3
    elif s == 'container_,_held':    return 'Cont:held', 4
    elif s == 'neither_,_touched':   return 'Neither:touched', 5
    elif s == 'neither_,_held':      return 'Neither:held', 6
    elif   s == 'None':              return s, -1
    else:
        print(f'Weird tool type label is {s}')
        pdb.set_trace()


def draw_hand_mask(im, draw, bbox, side, contact, grasp, mask, width, height, font, scale, use_simple):

    # parse
    side, side_idx = parseSide(side)
    contact, _ = parseState(contact)
    grasp, _   = parseGraspType(grasp)

    h_mask = mask[:, :, 0]
    h_mask = Image.fromarray(h_mask, mode='L')


    # bbox, mask
    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.bitmap((0, 0), h_mask, fill=hand_rgba[side_idx])
    pmask.rectangle(bbox, outline=hand_rgb[side_idx], width=4*scale, fill=None)
    im.paste(mask, (0,0), mask)
    # pdb.set_trace()
    
    # text
    
    if not use_simple:
        if contact == 'NoC':
            txt1 = contact
        else:
            txt1 = f'{contact},{grasp}'
        txt2 = f'{side}'
        txt1_width, txt1_height = draw.textsize(txt1, font)

        draw = ImageDraw.Draw(im)
        draw.rectangle([bbox[0], max(0, bbox[1]-43), bbox[0]+txt1_width+14, max(0, bbox[1]-41)+41], fill=(255, 255, 255), outline=hand_rgb[side_idx], width=4)
        draw.text((bbox[0]+8, max(0, bbox[1]-43)), txt1, font=font, fill=txt_color) # 
        draw.text((bbox[0]+6, max(0, bbox[3])), txt2, font=font, fill=hand_rgb[side_idx]) # 

    return im


def draw_firstobj_mask(im, draw, bbox, touch, mask, width, height, font, scale, use_simple):

    # parse
    touch, _   = parseTouchType(touch)

    fo_mask = mask[:, :, 0]
    fo_mask = Image.fromarray(fo_mask, mode='L')


    # bbox, mask
    mask  = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
   
    pmask.bitmap((0, 0), fo_mask, fill=firstobj_rgba)
    pmask.rectangle(bbox, outline=firstobj_rgb, width=4*scale, fill=None)
    im.paste(mask, (0,0), mask)
    
    # text
    
    if not use_simple:
        txt = touch
        draw = ImageDraw.Draw(im)
        txt_width, txt_height = draw.textsize(txt, font)
        draw.rectangle([bbox[0], max(0, bbox[1]-43), bbox[0]+txt_width+14, max(0, bbox[1]-43)+43], fill=(255, 255, 255), outline=firstobj_rgb, width=4)
        draw.text((bbox[0]+8, max(0, bbox[1]-41)), txt, font=font, fill=txt_color) # 
        # print(txt_width, txt_height)
    return im


def draw_secondobj_mask(im, draw, bbox, mask, width, height, font, scale, use_simple):

    so_mask = mask[:, :, 0]
    so_mask = Image.fromarray(so_mask, mode='L')

    # bbox, mask
    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.bitmap((0, 0), so_mask, fill=secondobj_rgba)
    pmask.rectangle(bbox, outline=secondobj_rgb, width=4*scale, fill=None)
    im.paste(mask, (0,0), mask)
    
    # text
    draw = ImageDraw.Draw(im)
    # draw.rectangle([bbox[0], max(0, bbox[1]-30), bbox[0]+62, max(0, bbox[1]-30)+30], fill=(255, 255, 255), outline=secondobj_rgb, width=4)

    return im


def draw_line_point(draw, center1, center2, color1, color2, scale):
    
    draw.line([center1, center2], fill=color1, width=4*scale)
    x, y = center1[0], center1[1]
    r=7 * scale
    draw.ellipse((x-r, y-r, x+r, y+r), fill=color1)
    x, y = center2[0], center2[1]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=color2)

def calculate_center(bb):
    '''bb: xyxy
    '''
    return (int((bb[0] + bb[2])/2), int((bb[1] + bb[3])/2) )



def vis_per_image(im, preds, filename, masks_dir, font_path='./times_b.ttf', use_simple=False):
    '''Given im and its preds, plot preds on im using PIL which has the opacity effect.
    '''

    # convert im to PIL format
    
    im = im[:,:,::-1]
    im = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(im)
    
    width, height = im.size 
    scale = max(width // 500, 1)
    font = ImageFont.truetype(font_path, size=35)

    for idx, p in enumerate(preds):
        h_bbox    = [ float(x) for x in p['hand_bbox']]
        h_side    = p['hand_side']
        _, side_idx = parseSide(h_side)
        h_contact = p['contact_state']
        h_grasp   = p['grasp']
      

        fo_bbox   = p['obj_bbox']
        so_bbox   = p['second_obj_bbox']

        # h_mask_path     = os.path.join(masks_dir, f'2_{idx}_{filename}.jpg')
        # fo_mask_path    = os.path.join(masks_dir, f'3_{idx}_{filename}.jpg')
        # so_mask_path    = os.path.join(masks_dir, f'5_{idx}_{filename}.jpg')
        h_mask_path     = os.path.join(masks_dir, f'2_{idx}_{filename}.png')
        fo_mask_path    = os.path.join(masks_dir, f'3_{idx}_{filename}.png')
        so_mask_path    = os.path.join(masks_dir, f'5_{idx}_{filename}.png')
        # h_mask_path     = os.path.join(masks_dir, f'2_{idx}_{filename}')
        # fo_mask_path    = os.path.join(masks_dir, f'3_{idx}_{filename}')
        # so_mask_path    = os.path.join(masks_dir, f'5_{idx}_{filename}')
        
        if os.path.exists(h_mask_path):
            h_mask = cv2.imread(h_mask_path)
        else:
            print(f'mask not exist: {h_mask_path}')
        
        

        # pdb.set_trace()
        # draw hand
        if h_bbox is not None:
            h_center = calculate_center(h_bbox)

            # draw first obj
            if fo_bbox is not None:
                fo_bbox   = [ float(x) for x in fo_bbox]
                fo_touch  = p['obj_touch']
                if os.path.exists(fo_mask_path):
                    fo_mask = cv2.imread(fo_mask_path)
                else:
                    print(f'mask not exist: {fo_mask_path}')
                #
                fo_center = calculate_center(fo_bbox)

                # draw second obj
                if so_bbox is not None:
                    so_bbox   = [ float(x) for x in so_bbox ]
                    if os.path.exists(so_mask_path):
                        so_mask = cv2.imread(so_mask_path)
                    else:
                        print(f'mask not exist: {so_mask_path}')
                    #
                    so_center = calculate_center(so_bbox)
                    

        # plot in order
        if so_bbox is not None:
            im = draw_secondobj_mask(im, draw, so_bbox, so_mask, width, height, font, scale, use_simple)
            
        if fo_bbox is not None:
            im = draw_firstobj_mask(im, draw, fo_bbox, fo_touch, fo_mask, width, height, font, scale, use_simple)
            if so_bbox is not None:
                draw_line_point(draw, fo_center, so_center, firstobj_rgb, secondobj_rgb, scale)

        if h_bbox is not None:
            im = draw_hand_mask(im, draw, h_bbox, h_side, h_contact, h_grasp, h_mask, width, height, font, scale, use_simple)
            if fo_bbox is not None:
                draw_line_point(draw, h_center, fo_center, hand_rgb[side_idx], firstobj_rgb, scale)


    return im


