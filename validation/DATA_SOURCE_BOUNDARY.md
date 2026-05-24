# Data Source Boundary

This project separates three annotation-related data types.

## `human_annotation_real`

Real responses submitted by human annotators. This data type may exist only when a real completed file is provided by the user or study organizer. Scripts require `data_source=human_annotation_real` before producing human-annotation analysis outputs.

## `synthetic_annotator_model`

Program-generated synthetic annotator responses. These responses are used only for pipeline sanity checks, power-analysis rehearsal, negative controls, and script testing. They are not human data and must not be described as player or expert evidence.

## `annotation_protocol_only`

Forms, codebooks, and synthetic session descriptors prepared for future annotation. These materials are not validation evidence by themselves.

## Forbidden Claims Unless Real Data Exist

- human annotation completed
- player study completed
- survey validated
- expert labels confirmed
- validated by participants
- construct validity established

## Allowed Claims in the Current State

- Chinese annotation protocol
- pilot annotation protocol
- future human validation
- synthetic annotator sanity check
- annotation-pipeline simulation
- pre-empirical construct-validity preparation
