division_prompt = """
you are a professional manga script writer.
your job is to divide this scenario script into a list of elements.

<Task>
Your focus is to divide the script into a list of elements. Elements are consisted of description and dialogue.
</Task>


<Definition>
Elements are the following
- Description:
  - The description is a description of the scene.
  - The description should be multiple sentences if the scene is the same.
- Dialogue:
  - The dialogue is a speech of a character.
  - The part which contains "「" and "」" is the dialogue.
  - In the dialogue, "M" letter never appears before "「".
  - Dialogue contains the speaker's name which appears before "「".
- Monologue:
  - The monologue is a inner thought of a character.
  - The monologue is not a dialogue.
  - The part which contains "「" and "」" or "『" and "』" is the monologue.
  - In the monologue, "M" letter appears before "「" or "『".
  - Monologue contains the speaker's name which appears before "M" letter.
</Definition>


<Guidelines>
Please follow these guidelines to create a list of elements which are described in the Definition abobve:

1. DO NOT make up the description or the dialogue.
2. Exacly extract the description and the dialogue from the given script.
3. Preserve the order of apeearances of the description and the dialogue in the given script.
4. The descriiption MUST NOT be extracted together from before the dialogue and after the dialogue.
5. DO NOT miss any description or dialogue. That means the concatenated output list should be exactly the same as the given script.
6. The output should be a valid JSON array and DO NOT answer it in markdown format. Answer it in the row text.
7. If the element is a monologue, the speaker's name MUST BE extracted.
8. If the element is a dialogue, the speaker's name MUST BE extracted.
</Guidelines>

<Output Format>
Present the list in the following format:
```
[
    {
        "content": "The description or dialogue here. Don't make up anything. Just extract it from the given script."
        "type": "description" or "dialogue" or "monologue"
        "speaker": "The speaker's name. If the element is a monologue, the speaker's name MUST BE extracted. If the element is a dialogue, the speaker's name MUST BE extracted. If the element is a description, the speaker's name MUST BE empty."
    },
    {
        "content": "The description or dialogue here. Don't make up anything. Just extract it from the given script."
        "type": "description" or "dialogue" or "monologue"
        "speaker": "The speaker's name. If the element is a monologue, the speaker's name MUST BE extracted. If the element is a dialogue, the speaker's name MUST BE extracted. If the element is a description, the speaker's name MUST BE empty."
    },
]
```
</Output Format>

"""


input_example = """
無関心な人々を見て、諦めたようにうなだれる蝶子。
蝶子「この世にあたしの味方なんていないんだ…。」

すれ違いざまに蝶子の言葉を聞いた椿。
気になって立ち止まる。


椿「あんた、味方が欲しいの？」

急に声をかけられ、驚いて振り返る蝶子。


（椿の容姿）
細身の黒いスリーピーススーツ、金髪の美青年。二十歳。色白。前髪長め。蝶子との身長差は頭一つ分くらい。

凛とした出で立ちでこちらを見つめる椿。


蝶子M「カッコイイ人…」
蝶子、椿の容姿に目を奪われる。


慎二「バカ！こっち来い！」

腕を思い切り引っ張って、慎二が蝶子を自分の方へ引き寄せる。


ホテル前で慎二に腕を掴まれ、痛そうに顔をしかめる蝶子。
椿、前髪をかき上げる。

椿「俺が味方になってあげよっか。」

蝶子M「（例え気紛れだったとしても、その言葉に、あの時のあたしがどれだけ救われたかなんて、王子様にはわからないだろうな。）」


蝶子焦ったように。
蝶子「お願いッ、たすけ…」

慎二、舌打ちしながら蝶子の言葉を遮る。
慎二「俺以外の男と話すな！無視しろよ。」
慎二に怒鳴られ、身をすくめる蝶子。

"""

output_example = """
[
    {
        "content": "無関心な人々を見て、諦めたようにうなだれる蝶子。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "蝶子「この世にあたしの味方なんていないんだ…。」",
        "type": "dialogue"
        "speaker": "蝶子"
    },
    {
        "content": "すれ違いざまに蝶子の言葉を聞いた椿。気になって立ち止まる。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "椿「あんた、味方が欲しいの？」",
        "type": "dialogue"
        "speaker": "椿"
    },
    {
        "content": "急に声をかけられ、驚いて振り返る蝶子。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "（椿の容姿）細身の黒いスリーピーススーツ、金髪の美青年。二十歳。色白。前髪長め。蝶子との身長差は頭一つ分くらい。凛とした出で立ちでこちらを見つめる椿。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "蝶子M「カッコイイ人…」",
        "type": "monologue"
        "speaker": "蝶子"
    },
    {
        "content": "蝶子、椿の容姿に目を奪われる。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "慎二「バカ！こっち来い！」",
        "type": "dialogue"
        "speaker": "慎二"
    },
    {
        "content": "腕を思い切り引っ張って、慎二が蝶子を自分の方へ引き寄せる。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "ホテル前で慎二に腕を掴まされ、痛そうに顔をしかめる蝶子。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "椿、前髪をかき上げる。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "椿「俺が味方になってあげよっか。」",
        "type": "dialogue"
        "speaker": "椿"
    },
    {
        "content": "蝶子M「（例え気紛れだったとしても、その言葉に、あの時のあたしがどれだけ救われたかなんて、王子様にはわからないだろうな。）」",
        "type": "monologue"
        "speaker": "蝶子"
    },
    {
        "content": "蝶子焦ったように。",
        "type": "description"
        "speaker": ""
    },
    {
        "content": "蝶子「お願いッ、たすけ…」",
        "type": "dialogue"
        "speaker": "蝶子"
    },
    {
        "content": "慎二、舌打ちしながら蝶子の言葉を遮る。",
        "type": "dialogue"
        "speaker": "慎二"
    },
    {
        "content": "慎二「俺以外の男と話すな！無視しろよ。」",
        "type": "dialogue"
        "speaker": "慎二"
    },
    {
        "content": "慎二に怒鳴られ、身をすくめる蝶子。",
        "type": "description"
        "speaker": ""

    }
]

"""

panel_prompt = """
you are a professional manga script writer.
your job is to divide the given list of elements into panels.

<Task>
Your focus is to divide the given list of elements into panels.
</Task>

<Definition>
element:
- an element is a dict object with the following keys:
    - content: a string of the content of the element.
    - type: a string of the type of the element. description, dialogue, or monologue.
    - speaker: a string of the speaker of the element. If the element is a description, the speaker is an empty string.
panel:
- a panel is a chunk of elements.
- a panel contains at least one element.
- a panel MUST NOT contain two or more elements whose type is description.
- a panel may contain multiple elements whose type is dialogue or monologue.
</Definition>

<Guidelines>
Please follow these guidelines to divide the given list of elements into panels:

1. Each panel MUST NOT contain more than three elements regardless of the type of the elements.
2. Each panel MUST NOT contain two or more elements whose type is description.
3. Each panel MUST contain the elements which is inferred to be spoken or occured in the same scene, time, place , or situation.
4. If there are more than three elements in a panel, divide the panel into smaller panels to satisfy the above rules.
5. DO NOT miss any element. DO NOT make up any element.
6. The output should be a valid JSON array and DO NOT answer it in markdown format. Answer it in the row text.
</Guidelines>

<Output Format>
Output MUST BE a list of panels.
Here is the format of the output:
```
[
    [
        {element1},
        {element2}
    ],
    [
        {element3},
    ],
    [
        {element4},
        {element5},
        {element6},
    ]
]
```
</Output Format>
"""

panel_example_input = """
[
    {
        "content": "無関心な人々を見て、諦めたようにうなだれる蝶子。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "蝶子「この世にあたしの味方なんていないんだ…。」",
        "type": "dialogue",
        "speaker": "蝶子"
    },
    {
        "content": "すれ違いざまに蝶子の言葉を聞いた椿。気になって立ち止まる。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "椿「あんた、味方が欲しいの？」",
        "type": "dialogue",
        "speaker": "椿"
    },
    {
        "content": "急に声をかけられ、驚いて振り返る蝶子。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "（椿の容姿）細身の黒いスリーピーススーツ、金髪の美青年。二十歳。色白。前髪長め。蝶子との身長差は頭一つ分くらい。凛とした出で立ちでこちらを見つめる椿。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "蝶子M「カッコイイ人…」",
        "type": "monologue",
        "speaker": "蝶子"
    },
    {
        "content": "蝶子、椿の容姿に目を奪われる。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "慎二「バカ！こっち来い！」",
        "type": "dialogue",
        "speaker": "慎二"
    },
    {
        "content": "腕を思い切り引っ張って、慎二が蝶子を自分の方へ引き寄せる。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "ホテル前で慎二に腕を掴まれ、痛そうに顔をしかめる蝶子。椿、前髪をかき上げる。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "椿「俺が味方になってあげよっか。」",
        "type": "dialogue",
        "speaker": "椿"
    },
    {
        "content": "蝶子M「（例え気紛れだったとしても、その言葉に、あの時のあたしがどれだけ救われたかなんて、王子様にはわからないだろうな。）」",
        "type": "monologue",
        "speaker": "蝶子"
    },
    {
        "content": "蝶子焦ったように。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "蝶子「お願いッ、たすけ…」",
        "type": "dialogue",
        "speaker": "蝶子"
    },
    {
        "content": "慎二、舌打ちしながら蝶子の言葉を遮る。",
        "type": "description",
        "speaker": ""
    },
    {
        "content": "慎二「俺以外の男と話すな！無視しろよ。」",
        "type": "dialogue",
        "speaker": "慎二"
    },
    {
        "content": "慎二に怒鳴られ、身をすくめる蝶子。",
        "type": "description",
        "speaker": ""
    },
"""

panel_example_output = """
[
    [
        {
            "content": "無関心な人々を見て、諦めたようにうなだれる蝶子。",
            "type": "description",
            "speaker": ""
        },
        {
            "content": "蝶子「この世にあたしの味方なんていないんだ…。」",
            "type": "dialogue",
            "speaker": "蝶子"
        },
    ],
    [
        {
            "content": "すれ違いざまに蝶子の言葉を聞いた椿。気になって立ち止まる。",
            "type": "description",
            "speaker": ""
        },
        {
            "content": "椿「あんた、味方が欲しいの？」",
            "type": "dialogue",
            "speaker": "椿"
        },
    ],
    [
        {
            "content": "急に声をかけられ、驚いて振り返る蝶子。",
            "type": "description",
            "speaker": ""
        },
    ],
    [
        {
            "content": "（椿の容姿）細身の黒いスリーピーススーツ、金髪の美青年。二十歳。色白。前髪長め。蝶子との身長差は頭一つ分くらい。凛とした出で立ちでこちらを見つめる椿。",
            "type": "description",
            "speaker": ""
        },
        {
            "content": "蝶子M「カッコイイ人…」",
            "type": "monologue",
            "speaker": "蝶子"
        },
    ], 
    [
        {
            "content": "蝶子、椿の容姿に目を奪われる。",
            "type": "description",
            "speaker": ""
        },
    ],
    [
        {
            "content": "慎二「バカ！こっち来い！」",
            "type": "dialogue",
            "speaker": "慎二"
        },
        {
            "content": "腕を思い切り引っ張って、慎二が蝶子を自分の方へ引き寄せる。",
            "type": "description",
            "speaker": ""
        },
    ],
    [
        {
            "content": "ホテル前で慎二に腕を掴まれ、痛そうに顔をしかめる蝶子。椿、前髪をかき上げる。",
            "type": "description",
            "speaker": ""
        },
        {
            "content": "椿「俺が味方になってあげよっか。」",
            "type": "dialogue",
            "speaker": "椿"
        },
        {
            "content": "蝶子M「（例え気紛れだったとしても、その言葉に、あの時のあたしがどれだけ救われたかなんて、王子様にはわからないだろうな。）」",
            "type": "monologue",
            "speaker": "蝶子"
        },
    ],
    [
        {
            "content": "蝶子焦ったように。",
            "type": "description",
            "speaker": ""
        },
        {
            "content": "蝶子「お願いッ、たすけ…」",
            "type": "dialogue",
            "speaker": "蝶子"
        },
    ],
    [
        {
            "content": "慎二、舌打ちしながら蝶子の言葉を遮る。",
            "type": "description",
            "speaker": ""
        },
        {
            "content": "慎二「俺以外の男と話すな！無視しろよ。」",
            "type": "dialogue",
            "speaker": "慎二"
        },
    ],
    [
        {
            "content": "慎二に怒鳴られ、身をすくめる蝶子。",
            "type": "description",
            "speaker": ""
        },
    ]
"""

richfy_prompt = """
you are a professional manga script writer.
your job is to modify or add description of the panel.


<Task>
Your focus is to modify or add description of the panel.
</Task>

<Definition>
element:
- an element is a dict object with the following keys:
    - content: a string of the content of the element.
    - type: a string of the type of the element. description, dialogue, or monologue.
    - speaker: a string of the speaker of the element. If the element is a description, the speaker is an empty string.
panel:
- a panel is a chunk of elements.
- a panel contains at least one element.
- a panel MUST NOT contain two or more elements whose type is description.
- a panel may contain multiple elements whose type is dialogue or monologue.
</Definition>

<Guidelines>
Please follow these guidelines to modify or add description of the panel:

1. If the panel already contains an element whose type is description, modify the description of the element. If the subject is missing, add subject to the corresponding verb based on the context. Modification should be as minimal as possible.
2. If the panel does not contain an element whose type is description, add a description to the panel. Generate a single sentence description based on the context. Set the type to description and the speaker to an empty string.
3. There MUST BE one description in each panel.
4. Generated description should be the first element of the panel. (Insert it at the beginning of the panel)
5. It is totally OK if the number of elements in the panel is more than three by this modification.
6. DO NOT change any elements whose type is dialogue or monologue.
7. DO NOT change the order of the elements in the panel.
8. The output should be a valid JSON array and DO NOT answer it in markdown format. Answer it in the row text.
</Guidelines>

<Output Format>
Output MUST BE a list of panels.
Here is the format of the output:
```
[
    [
        {element1},
        {element2}
    ],
]
```
"""