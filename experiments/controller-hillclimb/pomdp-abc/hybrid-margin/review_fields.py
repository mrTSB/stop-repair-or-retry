"""Read completed JSON fields from a truncated review without inventing missing fields."""

import json
import re


def fields(text):
    decoder=json.JSONDecoder(strict=False)
    text=re.sub(r'\\(?!["\\/bfnrtu])',lambda _: '\\\\',text)
    for start in re.finditer(r'\{',text):
        position=start.end();result={};complete=False
        while True:
            while position<len(text) and text[position].isspace():position+=1
            if position<len(text) and text[position]=='}':
                complete=True;break
            try:
                key,end=decoder.raw_decode(text,position)
                if not isinstance(key,str):break
                position=end
                while position<len(text) and text[position].isspace():position+=1
                if position>=len(text) or text[position]!=':':break
                position+=1
                while position<len(text) and text[position].isspace():position+=1
                value,end=decoder.raw_decode(text,position)
            except json.JSONDecodeError:
                break
            result[key]=value;position=end
            while position<len(text) and text[position].isspace():position+=1
            if position<len(text) and text[position]==',':position+=1
            elif position<len(text) and text[position]=='}':complete=True;break
            else:break
        if result:
            return result,complete
    return {},False


def parse(text):
    result,complete=fields(text)
    p=result.get('p_correct');action=result.get('action')
    valid=type(p) in (int,float) and 0<=p<=1 and action in ('stop','repair','retry')
    return result,bool(valid),complete
