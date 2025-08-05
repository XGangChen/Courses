#!/usr/bin/env python

import sys
from PyTexturePacker import Packer
from plistlib import load 
import re
import shutil
import pprint

def packCharacters( src, dest ):
    packer = Packer.create(max_width=2048, max_height=2048, shape_padding=4, trim_mode=1, reduce_border_artifacts=True, bg_color=0xffffff00)
    packer.pack( src, dest )

def nameToFields( fname ):
    p = re.compile(r'^\W*(?P<name>[A-Za-z0-9]+)[\W_]+(?P<obj>[A-Za-z0-9]+)[\W_]*(?P<tags>[^.]*)[.](png|jpg|jpeg)$')
    p = re.compile(r'^\W*(?P<name>[A-Za-z0-9]+)[\W_]+(?P<obj>[A-Za-z0-9]+)[\W_]*(?P<tags>[^.]*)[.](png|jpg|jpeg)$')
    m = p.match( fname )

#    print("Trying to match", fname)
    if m:
#        print("Match found fname", fname, "name", m.group('name'), "obj", m.group('obj'), "tags", m.group('tags')  )
        if m.group('name'):
            name = m.group('name').lower()
        else:
            name = "NAMEERROR"

        if m.group('obj'):
            obj = m.group('obj').lower()
        else:
            obj = "OBJERROR"

        if m.group('tags'):
            sp = re.compile(r'\W+')
            tags = sp.split( m.group( 'tags' ).lower() )
            #print('tags', tags)
        else:
            tags = []
        
        result =  ( name, obj, tags )
    else:
        result = None
    return result

def addFrame( root, frame ):
    obj = frame[1]
    print("Adding frame of type", obj )
    print("KinChain", root )
    robj, frames, prop, children = root
    print("addFrame", robj, frames, prop, children )
    if obj == robj:
        frames.append( frame )
    else:
        for c in children:
            addFrame( c, frame )
    return root

def appendToCSS( kinChain, pFile, rdir ):
    with open(pFile, "rb") as f:
        pd = load( f )

    frames = pd['frames']
    md = pd['metadata']

    #res = nameToFields('jacky-leftarm-up-down.jpeg')
    #res = nameToFields(' ProfJB_Head.png')

    for f in frames:
        print("Frame", f )
        res = nameToFields( f )
        print('res', res )
        print("rotated", frames[f]['rotated'] )

        id = res[0] + "_" + res[1]
        for t in res[2]:
            id = id + "_" + t
        print('id', id)
        print(frames[f])
        coords = re.compile(r' *{ *{ *-? *(?P<px>[0-9.]+) *, *-? *(?P<py>[0-9.]+) *} *, *{ *(?P<width>[0-9.]+) *, *(?P<height>[0-9.]+) *} *}')
        m = coords.match(frames[f]['frame'] )
        if m:
            px = int( m.group('px') )
            py = int( m.group('py') )
            width = int( m.group('width') )
            height = int( m.group('height') )
            print('texture', px, py, width, height )


        srect = re.compile(r' *{ *{ *-? *(?P<sx>[0-9.]+) *, *-? *(?P<sy>[0-9.]+) *} *, *{ *(?P<width>[0-9.]+) *, *(?P<height>[0-9.]+) *} *}')
        m = srect.match(frames[f]['sourceColorRect'] )
        if m:
            sx = int( m.group('sx') )
            sy = int( m.group('sy') )
            swidth = int( m.group('width') )
            sheight = int( m.group('height') )
            print('source rect', sx, sy, swidth, sheight )

            fileName = rdir + "/" + md['textureFileName']
            addFrame( kinChain, (res[0], res[1], res[2], id, width, height, frames[f]['rotated'], px, py, sx, sy, swidth, sheight, fileName ) )
    return kinChain

def createHTMLSource( kinRoot, cx=0, cy=0 ):
    obj, frames, prop, children = kinRoot

    css = ""
    js = ""
    htmlRoot = [ f"{obj}", [ f"cls-{obj}", "default" ], [] ]
    for f in frames:
        name, obj, tags, id, width, height, rotated, px, py, sx, sy, swidth, sheight, fileName = f
        if rotated:
            rot = " rotate(-90deg)"
            width,height = height,width
        else:
            rot = ""

        ps = "\n".join(prop)
        
        cssRec = f"""
#{id} {{
    width: {width - 2}px;
    height: {height - 2}px;
    position: absolute;
    background: url("{fileName}") -{px + 1}px -{py + 1}px;
    transform: translate( {sx - cx}px, {sy - cy}px){rot};
    {ps}
}}

#bone-{id} {{
    transform-origin: 50% 50%;
    position: absolute;
}}
"""
        css = css + cssRec

        ctags = ",".join(tags)
        if ctags == "":
            ctags = "default"

        htmlOption = [ [ f"bone-{id}", [ f"cls-bone-{id}", "default"], [ 
                        [ f"{id}", [ f"cls-{id}", f"{ctags}" ], [] ] ] ] ]
        htmlRoot[2] = htmlRoot[2] + htmlOption

        jsRec = f"""
function create_{id} (node) {{
    node.append
}}        
"""

        for c in children:
            cssC, htmlNode, jsC = createHTMLSource( c, sx, sy )
            css = css + cssC
            html = htmlOption[-1][-1].append(htmlNode)
            js = js + jsC

    return css, htmlRoot, js

# function createAnimation( container, character, mode, anim )
#

def createJSTree( name, htmlTree, level = 0 ):
    id, tags, children = htmlTree
    
    ts = ""
    for t in tags:
        if ( len(ts) > 0 ):
            ts = ts + ", "
        ts = ts + f'"{t}"'

    js = """
{lvl}{{  "id": "{id}", "tags":[{ts}], 
{lvl}   "children": 
{lvl}   [ 
    """.format(id=id, ts=ts, lvl = "   " * level )

    first = True
    for c in children:
        if ( not first ):
            js = js + ","
        js = js + "\n" + "   " * (level + 1)
        js = js + createJSTree(name, c, level + 2)
        first = False
        
    js = js + "\n" + "   " * (level + 1) + "]\n" + "   " * level + "}"
    return js

def main( argv = None ):
    if argv is None:
        argv = sys.argv[1:]
    #packCharacters( argv[0], argv[1] )

    profJBKinChains = [ "trunk", [ ], [ "z-index:10;"],
        [ 
            [ "head", [ ], [ "z-index:5;"],
                [   
                    [ "eyes", [ ], [  "z-index:5;" ], [ ] ], 
                    [ "mouth", [ ], [  "z-index:5;" ], [ ] ], 
                ],
            ],
            [ "leftarm", [ ], [  "z-index:5;" ], 
                [ 
                    [ "leftelbow", [ ], [ ], [ ] ],
                    [ "leftwrist", [ ], [ ], [ ] ] 
                ] 
            ], 
            [ "rightarm", [ ], [  "z-index:5;" ],
                [ 
                    [ "rightelbow", [ ], [ ], [ ] ],
                    [ "rightwrist", [ ], [ ], [ ] ] 
                ] 
            ],
        ]
    ] 
    
    profJBKinChains = appendToCSS(profJBKinChains, "prof_jb_0.plist", "../assets")
    profJBkinChains = appendToCSS(profJBKinChains, "prof_jb_1.plist", "../assets")
    
    css, htmlTree, js = createHTMLSource( profJBKinChains )

    with open('prof_jb.css','w') as f:
        f.write(css)

    with open('prof_jb.html','w') as f:
        f.write("var profJB = ")
        f.write(pprint.pformat(htmlTree).replace("'", '"'))
        f.write(";\n")

    jsTree = createJSTree( "prof_jb", htmlTree )
    with open('prof_jb_test.js','w') as f:
        f.write( "var ProfJB = \n")
        f.write( jsTree )
        f.write(";\n")

if __name__ == "__main__":
    main( [ '../../Lecture-VN/Resources/ProfJB/', "prof_jb_%d"] )
