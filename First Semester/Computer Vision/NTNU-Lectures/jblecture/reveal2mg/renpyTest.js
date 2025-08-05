const parser = require("./renpy.js")

tests = [   "'xxyyx cvcvc xxx'", 
            '  \n"This is the new lecture style"  \n', 
            '"""A\nmultiline\nstring"""', 
            "'''Another\r\ntest'''" ]
for(var i = 0; i < tests.length; i++ ) {
    try{
        console.log("Trying to parse |" + tests[i] + "|");
        parser.parse(tests[i]);
        console.log('Success');
    }
    catch(err) {
        console.log('Syntax Error');
        console.log(err);
        throw( err );
    }
} 

tests = [ 
"label test:\n" +
"       \"My message is clear!\"",
"label test2:\n" +
"   \"bob\" \"Hello\"\n" +
"  \t \"marry\" \"Hello Bob\"\n",
' \t jump "label123"',
"label test2:\n" +
"   \"bob\" \"Hello\"\n" +
"  # This is a comment\n   \n" +
"  \t jump \"Label1\"\n" +
"  \t jump Label1    \t \n \n   \n",

]

for(var i = 0; i < tests.length; i++ ) {
    try{
        console.log("Trying to parse |" + tests[i] + "|");
        parser.parse(tests[i]);
        console.log('Success');
    }
    catch(err) {
        console.log('Syntax Error');
        console.log(err);
        throw( err );
    }
} 

tests = [ 
    "pause 2\npause\npause 2.2",
    "show jb neutral with fade",
    "show msG happy arm raised\n   pause 0.5",
    "show jb neutral at center with fade",
    ]
    
    for(var i = 0; i < tests.length; i++ ) {
        try{
            console.log("Trying to parse |" + tests[i] + "|");
            parser.parse(tests[i]);
            console.log('Success');
        }
        catch(err) {
            console.log('Syntax Error');
            console.log(err);
            throw( err );
        }
    } 


tests = [ 
"scene bg neutral with fade\n",
"    window      show    ",
"window hide",
"    hide me\n",
];

for(var i = 0; i < tests.length; i++ ) {
    try{
        console.log("Trying to parse |" + tests[i] + "|");
        parser.parse(tests[i]);
        console.log('Success');
    }
    catch(err) {
        console.log('Syntax Error');
        console.log(err);
        throw( err );
    }
} 


tests = [
    "label start:\n"+
    "\n" + 
    "\"Sylvie\" \"Hi there! How was class?\"\n" +
    "\n" + 
    "\n" +         
    "\"Me\" \"Good...\"\n" + 
    "\n" + 
    "\"I can't bring myself to admit that it all went in one ear and out the other.\"\n"  + 
    "\n" + 
    "\"Me\" \"Are you going home now? Wanna walk back with me?\"\n" +
    "\n" + 
    "\"Sylvie\" \"Sure!\""
    ];
    
for(var i = 0; i < tests.length; i++ ) {
    try{
        console.log("Trying to parse |" + tests[i] + "|");
        parser.parse(tests[i]);
        console.log('Success');
    }
    catch(err) {
        console.log('Syntax Error');
        console.log(err);
        throw( err );
    }
} 
    
