import sys
import renpy_grammar

class Actions( object ):
    def make_say1(self, input, start, end, elements ):
        print( "make_say1(", len(elements), ")" )
        #print('SAY', elements[2].text, elements[3].text )
        return( elements[2], elements[3] )

    def make_jump( self, input, start, end, elements ):
        print("make_jump", input[start:end] )
        return elements[3]

    def make_label( self, input, start, end, elements ):
        #print("make_label", elements[3] )    
        return elements 

    def make_show( self, input, start, end, elements ):
        modifiers = elements[4].text    
        print("make_show", elements, modifiers )
        return elements 
 
    def make_position( self, input, start, end, elements ):
        position = elements[2]   
        print("make_position", elements, position )
        return position
 
    def make_effect( self, input, start, end, elements ):
        effect = elements[2]    
        print("make_effect", elements, effect )
        return effect
 
    def make_blank( self, input, start, end, elements ):
        print("make_blank", elements )    
        return None 

    def make_qstring( self, input, start, end, elements ):
        print("make_qstring", input[start:end], elements[1].text )    
        return elements[1].text 

    def make_string( self, input, start, end, elements ):
        print("make_string", input[start:end] )    
        return input[start:end]

    def make_pause( self, input, start, end, elements ):
        print("make_pause", input[start:end], elements[2].elements )
        if len(elements[2].elements) >= 2 and elements[2].elements[1] > 0:
            time = elements[2].elements[1]
        else:
            time = 0.0

        return "pause {0:5.2f}".format(time)

    def convertSpeech( self, node ):
        print('convertSpeech', len(node.elements), node.elements[1].text )
        return node.elements[1].text.strip()
    
    def convertID( self, node ):
        print('convertSpeech', len(node.elements), node.elements[1].text )
        return node.elements[1].text.strip()

    def make_num( self, input, start, end, elements ):
        print("make_num", input[start:end] )    
        return float( input[start:end] )


def main( argv = None ):
    if not argv:
        argv = sys.argv[1:]
    result = renpy_grammar.parse(text, actions=Actions() )
    print('result', result)

text = """
label start:

    pause 
    show me happy arm raised at center
    "Sylvie" "Hi there! How was class?"

    say "Me" 'Good...'

    show "me" at center

    "I can't bring myself to admit that it all went in one ear and out the other."

    "Me" \"\"\"
    Are you going home now? 
    Wanna walk back with me?
    \"\"\"

    "Sylvie" "Sure!"

    pause 1.2

    jump slide2
"""

text = "show me happy up at center with fade\n"
#text = """show me at center\n"""

if __name__ == "__main__":
    main()
