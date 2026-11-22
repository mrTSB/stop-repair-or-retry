"""Outcome-independent512-question paired TRAIN extension; no label reader."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import expanded_mentor_io as io
import science_auxiliary_data as auxiliary
import reserve_science_aux
import science_complete_actions as complete

SCHEMA='auxiliary-science512-plain7-Complete3-TRAIN-v1'
SEED=2026112315
COUNTS={'chemistry':256,'physics':256}
PLAIN=complete.pilot.protocol.PLAIN
PATHS=(*PLAIN,*complete.PATHS)
FORBIDDEN={'answer','answer_letter','rationale','explanation','solution'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def project(raw,wanted):
    """Decode only approved fields; skip reference values as opaque JSON bytes."""
    if wanted&FORBIDDEN or not wanted<={'uuid','field','discipline','question','options','id'}:
        raise ValueError('Reference or unknown field projection forbidden')
    def whitespace(i):
        while i<len(raw) and raw[i] in b' \n\r\t':i+=1
        return i
    def string_end(i):
        if raw[i]!=34:raise ValueError('Expected JSON string')
        i+=1
        while i<len(raw):
            if raw[i]==92:i+=2;continue
            if raw[i]==34:return i+1
            i+=1
        raise ValueError('Unterminated JSON string')
    def value_end(i):
        if raw[i]==34:return string_end(i)
        if raw[i] in (91,123):
            stack=[raw[i]];i+=1
            while stack:
                if raw[i]==34:i=string_end(i);continue
                if raw[i] in (91,123):stack.append(raw[i])
                elif raw[i] in (93,125):
                    if (stack.pop(),raw[i]) not in ((91,93),(123,125)):raise ValueError('Invalid JSON container')
                i+=1
            return i
        while i<len(raw) and raw[i] not in b',} \n\r\t':i+=1
        return i
    i=whitespace(0)
    if raw[i]!=123:raise ValueError('Expected source object')
    i+=1;out={};keys=set()
    while True:
        i=whitespace(i)
        if raw[i]==125:break
        end=string_end(i);key=json.loads(raw[i:end]);i=whitespace(end)
        if key in keys or raw[i]!=58:raise ValueError('Duplicate source key or missing colon')
        keys.add(key);start=whitespace(i+1);end=value_end(start)
        if key in wanted:out[key]=json.loads(raw[start:end])
        i=whitespace(end)
        if raw[i]==44:i+=1
        elif raw[i]!=125:raise ValueError('Missing source delimiter')
    if wanted-set(out):raise ValueError('Missing requested source field')
    return out


def choose(subjects):
    if len(subjects)!=1440 or Counter(subjects.values())!={'chemistry':720,'physics':720}:
        raise ValueError('Use the exact balanced1440 auxiliary TRAIN pool')
    selected=[]
    for subject,n in COUNTS.items():
        group=sorted((qid for qid,s in subjects.items() if s==subject),
            key=lambda qid:hashlib.sha256(f'{SEED}:auxiliary-paired:{qid}'.encode()).digest())
        selected.extend({'id':qid,'subject':subject} for qid in group[:n])
    return sorted(selected,key=lambda item:hashlib.sha256(f'{SEED}:order:{item["id"]}'.encode()).digest())


def public_inputs(args):
    metadata=io.read(args.auxiliary/'metadata.json');seal=io.read(args.auxiliary/'provenance/seal-receipt.json')
    exclusion=io.read(args.auxiliary/'provenance/historical-exclusions.json');selection=io.read(args.data/'splits/selection.json')
    pin=io.read(args.cache/'pin.json');source=args.cache/'SuperGPQA-all.jsonl';recipe=io.read(args.observer/'recipe.json')
    if (seal['metadata_sha256']!=io.sha(args.auxiliary/'metadata.json')
        or seal['historical_exclusions_sha256']!=io.sha(args.auxiliary/'provenance/historical-exclusions.json')
        or seal['sealer_sha256']!=io.sha(auxiliary.__file__) or seal['protected_overlap']!=0
        or seal['train_questions']!=1440 or seal['calibration_questions']!=160
        or metadata['source_code_sha256']!=io.sha(reserve_science_aux.__file__)
        or metadata['source_sha256']!=io.sha(source) or seal['source_sha256']!=metadata['source_sha256']
        or (pin['repo'],pin['revision'])!=(metadata['repository'],metadata['revision'])
        or selection['science']['sha256']!=metadata['source_sha256']
        or metadata['protected_split_sha256']!={s:v['sha256'] for s,v in selection['splits'].items()}):
        raise ValueError('Changed auxiliary source, historical exclusions or project split reservation seals')
    train={'supergpqa-'+qid for qid in metadata['train_ids']};cal={'supergpqa-'+qid for qid in metadata['calibration_ids']}
    if (len(train)!=1440 or len(cal)!=160 or train&cal or set(exclusion['ids'])!=train|cal
        or exclusion['metadata_sha256']!=io.sha(args.auxiliary/'metadata.json')
        or recipe['data_sha256']!=io.sha(args.legacy_examples) or recipe['fold']!='full'
        or recipe['input_profile']!='full-options-openstax-v1'):
        raise ValueError('Changed exact TRAIN/calibration membership or old unseen encoder ancestry')
    with args.legacy_examples.open('rb') as stream:legacy={project(line,{'id'})['id'] for line in stream}
    if (train|cal)&legacy:raise ValueError('Auxiliary question seen by frozen old semantic encoder')
    subjects={}
    with source.open('rb') as stream:
        for raw in stream:
            header=project(raw,{'uuid','field','discipline'});qid='supergpqa-'+header['uuid']
            if qid in train:
                if qid in subjects or header['discipline']!='Science':raise ValueError('Duplicate or nonscience auxiliary TRAIN UUID')
                subjects[qid]=header['field'].casefold()
    chosen=choose(subjects);by_id={item['id']:item for item in chosen};public={}
    with source.open('rb') as stream:
        for raw in stream:
            qid='supergpqa-'+project(raw,{'uuid'})['uuid']
            if qid not in by_id:continue
            item=project(raw,{'question','options'});options=[v.strip() for v in item['options']]
            if not (4<=len(options)<=10 and all(options) and len(set(options))==len(options)):
                raise ValueError('Malformed fixed-cohort official choices; no quality replacement')
            public[qid]={'id':qid,'domain':'science','subject':by_id[qid]['subject'],
                'question':auxiliary.question({'question':item['question'],'options':options}),'answer':'',
                'source':metadata['repository']+'@'+metadata['revision']}
    if set(public)!=set(by_id):raise ValueError('Missing fixed-cohort public source row')
    files=[args.auxiliary/'metadata.json',args.auxiliary/'provenance/seal-receipt.json',
        args.auxiliary/'provenance/historical-exclusions.json',args.data/'splits/selection.json',
        args.cache/'pin.json',source,args.observer/'recipe.json',args.legacy_examples]
    evidence={str(p.resolve()):io.sha(p) for p in files}
    receipt={'files':evidence,'auxiliary_train_ids_sha256':fingerprint(sorted(train)),
        'excluded_calibration_ids_sha256':fingerprint(sorted(cal)),'legacy_encoder_training_ids_sha256':fingerprint(sorted(legacy)),
        'legacy_ids':len(legacy),'legacy_encoder_overlap':0,'project_split_overlap':0,
        'protected_split_sha256':metadata['protected_split_sha256'],
        'project_exclusion_basis':'Exact prior auxiliary sealer receipt bound to unchanged source/membership and project reservation hashes; no new protected split reads',
        'reference_fields_decoded':False,'reserved_question_files_opened':False,'auxiliary_encoder_features_allowed':False}
    return [public[item['id']] for item in chosen],chosen,receipt


def priority_receipt(path,blind_first=False):
    folder='blind-first-root-review-pilot-v1' if blind_first else 'strong-late-review-pilot-v1'
    schema='blind-first-root-review-TRAIN-v1' if blind_first else 'strong-late-review-timing-TRAIN-v1'
    expected=io.PROJECT/'runs/controller-semantic-followup/round2'/folder/'complete.json'
    if path.resolve()!=expected.resolve():raise ValueError('Require exact completed priority strong-late pilot')
    done=io.read(path);registration=io.read(path.parent/'registration.json');coverage=done['coverage']
    if (done['schema']!=schema or registration['schema']!=done['schema']
        or done['registration_sha256']!=io.sha(path.parent/'registration.json') or not done['training_only']
        or done['test_used'] or done['new_neural_calls'] or done['scored']):
        raise ValueError('Priority acquisition is not fully sealed')
    if blind_first:
        if coverage!={'questions':64,'logical_reviews':64,'first_guided':64,'second_repairs':64,
            'physical_teacher_phases':128,'physical_compact_calls':128,'physical_calls':256}:
            raise ValueError('Require complete256-call blind-first priority pilot')
    elif (coverage['questions']!=64 or coverage['logical_reviews']!=128 or coverage['compact_repairs']!=128
        or coverage['physical_calls']!=128+coverage['physical_review_phases']):
        raise ValueError('Require complete strong-late priority pilot')
    io.verify({str(path.parent/n):v for n,v in done['digests'].items()});io.verify(registration['source_files'])
    return {'path':str(path.resolve()),'sha256':io.sha(path),'registration_sha256':done['registration_sha256'],
        'source_files':registration['source_files']}
