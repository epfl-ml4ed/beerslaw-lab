��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ߠ
�
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameembedding_1/embeddings
�
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:d*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�d*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*.
shared_nameAdam/embedding_1/embeddings/m
�
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes

:d*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	�d*
dtype0
�
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*.
shared_nameAdam/embedding_1/embeddings/v
�
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes

:d*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	�d*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
loss
regularization_losses
trainable_variables
	variables
		keras_api


signatures
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
d
iter

beta_1

beta_2
	decay
learning_ratem=m>v?v@
 
 

0
1

0
1
�
regularization_losses
non_trainable_variables
trainable_variables
	variables
layer_metrics
 layer_regularization_losses

!layers
"metrics
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
regularization_losses
#non_trainable_variables
trainable_variables
	variables
$layer_metrics
%layer_regularization_losses

&layers
'metrics
 
 
 
�
regularization_losses
(non_trainable_variables
trainable_variables
	variables
)layer_metrics
*layer_regularization_losses

+layers
,metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
regularization_losses
-non_trainable_variables
trainable_variables
	variables
.layer_metrics
/layer_regularization_losses

0layers
1metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2

20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	4total
	5count
6	variables
7	keras_api
D
	8total
	9count
:
_fn_kwargs
;	variables
<	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

40
51

6	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

;	variables
��
VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
!serving_default_embedding_1_inputPlaceholder*'
_output_shapes
:���������d*
dtype0*
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_1_inputembedding_1/embeddingsdense_1/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_signature_wrapper_119089910
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_save_119090071
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_1/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/embedding_1/embeddings/mAdam/dense_1/kernel/mAdam/embedding_1/embeddings/vAdam/dense_1/kernel/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference__traced_restore_119090126��
�
�
+__inference_dense_1_layer_call_fn_119090003

inputs
unknown:	�d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_1190897992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089942

inputs8
&embedding_1_embedding_lookup_119089930:d9
&dense_1_matmul_readvariableop_resource:	�d
identity��dense_1/MatMul/ReadVariableOp�embedding_1/embedding_lookupu
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_119089930embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/119089930*+
_output_shapes
:���������d*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/119089930*+
_output_shapes
:���������d2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2)
'embedding_1/embedding_lookup/Identity_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����	  2
flatten_1/Const�
flatten_1/ReshapeReshape0embedding_1/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_1/MatMuly
dense_1/SoftmaxSoftmaxdense_1/MatMul:product:0*
T0*'
_output_shapes
:���������d2
dense_1/Softmaxt
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089882
embedding_1_input'
embedding_1_119089874:d$
dense_1_119089878:	�d
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_119089874*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_1190897792%
#embedding_1/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1190897892
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_119089878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_1190897992!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:Z V
'
_output_shapes
:���������d
+
_user_specified_nameembedding_1_input
�
�
/__inference_embedding_1_layer_call_fn_119089977

inputs
unknown:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_1190897792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089855

inputs'
embedding_1_119089847:d$
dense_1_119089851:	�d
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_119089847*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_1190897792%
#embedding_1/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1190897892
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_119089851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_1190897992!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
0__inference_sequential_1_layer_call_fn_119089951

inputs
unknown:d
	unknown_0:	�d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1190898042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
0__inference_sequential_1_layer_call_fn_119089811
embedding_1_input
unknown:d
	unknown_0:	�d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1190898042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:���������d
+
_user_specified_nameembedding_1_input
�
�
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089893
embedding_1_input'
embedding_1_119089885:d$
dense_1_119089889:	�d
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_119089885*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_1190897792%
#embedding_1/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1190897892
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_119089889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_1190897992!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:Z V
'
_output_shapes
:���������d
+
_user_specified_nameembedding_1_input
�
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_119089983

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_dense_1_layer_call_and_return_conditional_losses_119089996

inputs1
matmul_readvariableop_resource:	�d
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:���������d2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_sequential_1_layer_call_fn_119089871
embedding_1_input
unknown:d
	unknown_0:	�d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1190898552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:���������d
+
_user_specified_nameembedding_1_input
�
I
-__inference_flatten_1_layer_call_fn_119089988

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1190897892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
0__inference_sequential_1_layer_call_fn_119089960

inputs
unknown:d
	unknown_0:	�d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_1_layer_call_and_return_conditional_losses_1190898552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�*
�
"__inference__traced_save_119090071
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_1_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*h
_input_shapesW
U: :d:	�d: : : : : : : : : :d:	�d:d:	�d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d:%!

_output_shapes
:	�d:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d:%!

_output_shapes
:	�d:$ 

_output_shapes

:d:%!

_output_shapes
:	�d:

_output_shapes
: 
�C
�
%__inference__traced_restore_119090126
file_prefix9
'assignvariableop_embedding_1_embeddings:d4
!assignvariableop_1_dense_1_kernel:	�d&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: C
1assignvariableop_11_adam_embedding_1_embeddings_m:d<
)assignvariableop_12_adam_dense_1_kernel_m:	�dC
1assignvariableop_13_adam_embedding_1_embeddings_v:d<
)assignvariableop_14_adam_dense_1_kernel_v:	�d
identity_16��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp1assignvariableop_11_adam_embedding_1_embeddings_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_dense_1_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_adam_embedding_1_embeddings_vIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_1_kernel_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_15f
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_16�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_16Identity_16:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089926

inputs8
&embedding_1_embedding_lookup_119089914:d9
&dense_1_matmul_readvariableop_resource:	�d
identity��dense_1/MatMul/ReadVariableOp�embedding_1/embedding_lookupu
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather&embedding_1_embedding_lookup_119089914embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@embedding_1/embedding_lookup/119089914*+
_output_shapes
:���������d*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@embedding_1/embedding_lookup/119089914*+
_output_shapes
:���������d2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2)
'embedding_1/embedding_lookup/Identity_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����	  2
flatten_1/Const�
flatten_1/ReshapeReshape0embedding_1/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_1/MatMuly
dense_1/SoftmaxSoftmaxdense_1/MatMul:product:0*
T0*'
_output_shapes
:���������d2
dense_1/Softmaxt
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp^dense_1/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
'__inference_signature_wrapper_119089910
embedding_1_input
unknown:d
	unknown_0:	�d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__wrapped_model_1190897622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:���������d
+
_user_specified_nameembedding_1_input
�

�
J__inference_embedding_1_layer_call_and_return_conditional_losses_119089779

inputs,
embedding_lookup_119089773:d
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d2
Cast�
embedding_lookupResourceGatherembedding_lookup_119089773Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/119089773*+
_output_shapes
:���������d*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/119089773*+
_output_shapes
:���������d2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������d2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
$__inference__wrapped_model_119089762
embedding_1_inputE
3sequential_1_embedding_1_embedding_lookup_119089750:dF
3sequential_1_dense_1_matmul_readvariableop_resource:	�d
identity��*sequential_1/dense_1/MatMul/ReadVariableOp�)sequential_1/embedding_1/embedding_lookup�
sequential_1/embedding_1/CastCastembedding_1_input*

DstT0*

SrcT0*'
_output_shapes
:���������d2
sequential_1/embedding_1/Cast�
)sequential_1/embedding_1/embedding_lookupResourceGather3sequential_1_embedding_1_embedding_lookup_119089750!sequential_1/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*F
_class<
:8loc:@sequential_1/embedding_1/embedding_lookup/119089750*+
_output_shapes
:���������d*
dtype02+
)sequential_1/embedding_1/embedding_lookup�
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*F
_class<
:8loc:@sequential_1/embedding_1/embedding_lookup/119089750*+
_output_shapes
:���������d24
2sequential_1/embedding_1/embedding_lookup/Identity�
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d26
4sequential_1/embedding_1/embedding_lookup/Identity_1�
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����	  2
sequential_1/flatten_1/Const�
sequential_1/flatten_1/ReshapeReshape=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2 
sequential_1/flatten_1/Reshape�
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp�
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential_1/dense_1/MatMul�
sequential_1/dense_1/SoftmaxSoftmax%sequential_1/dense_1/MatMul:product:0*
T0*'
_output_shapes
:���������d2
sequential_1/dense_1/Softmax�
IdentityIdentity&sequential_1/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp+^sequential_1/dense_1/MatMul/ReadVariableOp*^sequential_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup:Z V
'
_output_shapes
:���������d
+
_user_specified_nameembedding_1_input
�
�
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089804

inputs'
embedding_1_119089780:d$
dense_1_119089800:	�d
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_119089780*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_embedding_1_layer_call_and_return_conditional_losses_1190897792%
#embedding_1/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_1190897892
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_119089800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_1190897992!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity�
NoOpNoOp ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_119089789

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
J__inference_embedding_1_layer_call_and_return_conditional_losses_119089970

inputs,
embedding_lookup_119089964:d
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d2
Cast�
embedding_lookupResourceGatherembedding_lookup_119089964Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/119089964*+
_output_shapes
:���������d*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/119089964*+
_output_shapes
:���������d2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������d2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
F__inference_dense_1_layer_call_and_return_conditional_losses_119089799

inputs1
matmul_readvariableop_resource:	�d
identity��MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:���������d2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
O
embedding_1_input:
#serving_default_embedding_1_input:0���������d;
dense_10
StatefulPartitionedCall:0���������dtensorflow/serving/predict:�K
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
loss
regularization_losses
trainable_variables
	variables
		keras_api


signatures
A_default_save_signature
*B&call_and_return_all_conditional_losses
C__call__"
_tf_keras_sequential
�

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
*D&call_and_return_all_conditional_losses
E__call__"
_tf_keras_layer
�
regularization_losses
trainable_variables
	variables
	keras_api
*F&call_and_return_all_conditional_losses
G__call__"
_tf_keras_layer
�

kernel
regularization_losses
trainable_variables
	variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"
_tf_keras_layer
w
iter

beta_1

beta_2
	decay
learning_ratem=m>v?v@"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
non_trainable_variables
trainable_variables
	variables
layer_metrics
 layer_regularization_losses

!layers
"metrics
C__call__
A_default_save_signature
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
,
Jserving_default"
signature_map
(:&d2embedding_1/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
regularization_losses
#non_trainable_variables
trainable_variables
	variables
$layer_metrics
%layer_regularization_losses

&layers
'metrics
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
(non_trainable_variables
trainable_variables
	variables
)layer_metrics
*layer_regularization_losses

+layers
,metrics
G__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
!:	�d2dense_1/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
regularization_losses
-non_trainable_variables
trainable_variables
	variables
.layer_metrics
/layer_regularization_losses

0layers
1metrics
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	4total
	5count
6	variables
7	keras_api"
_tf_keras_metric
^
	8total
	9count
:
_fn_kwargs
;	variables
<	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
40
51"
trackable_list_wrapper
-
6	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
-
;	variables"
_generic_user_object
-:+d2Adam/embedding_1/embeddings/m
&:$	�d2Adam/dense_1/kernel/m
-:+d2Adam/embedding_1/embeddings/v
&:$	�d2Adam/dense_1/kernel/v
�B�
$__inference__wrapped_model_119089762embedding_1_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089926
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089942
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089882
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089893�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_sequential_1_layer_call_fn_119089811
0__inference_sequential_1_layer_call_fn_119089951
0__inference_sequential_1_layer_call_fn_119089960
0__inference_sequential_1_layer_call_fn_119089871�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_embedding_1_layer_call_and_return_conditional_losses_119089970�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_embedding_1_layer_call_fn_119089977�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_flatten_1_layer_call_and_return_conditional_losses_119089983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_flatten_1_layer_call_fn_119089988�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_1_layer_call_and_return_conditional_losses_119089996�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_dense_1_layer_call_fn_119090003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_119089910embedding_1_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
$__inference__wrapped_model_119089762s:�7
0�-
+�(
embedding_1_input���������d
� "1�.
,
dense_1!�
dense_1���������d�
F__inference_dense_1_layer_call_and_return_conditional_losses_119089996\0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� ~
+__inference_dense_1_layer_call_fn_119090003O0�-
&�#
!�
inputs����������
� "����������d�
J__inference_embedding_1_layer_call_and_return_conditional_losses_119089970_/�,
%�"
 �
inputs���������d
� ")�&
�
0���������d
� �
/__inference_embedding_1_layer_call_fn_119089977R/�,
%�"
 �
inputs���������d
� "����������d�
H__inference_flatten_1_layer_call_and_return_conditional_losses_119089983]3�0
)�&
$�!
inputs���������d
� "&�#
�
0����������
� �
-__inference_flatten_1_layer_call_fn_119089988P3�0
)�&
$�!
inputs���������d
� "������������
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089882oB�?
8�5
+�(
embedding_1_input���������d
p 

 
� "%�"
�
0���������d
� �
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089893oB�?
8�5
+�(
embedding_1_input���������d
p

 
� "%�"
�
0���������d
� �
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089926d7�4
-�*
 �
inputs���������d
p 

 
� "%�"
�
0���������d
� �
K__inference_sequential_1_layer_call_and_return_conditional_losses_119089942d7�4
-�*
 �
inputs���������d
p

 
� "%�"
�
0���������d
� �
0__inference_sequential_1_layer_call_fn_119089811bB�?
8�5
+�(
embedding_1_input���������d
p 

 
� "����������d�
0__inference_sequential_1_layer_call_fn_119089871bB�?
8�5
+�(
embedding_1_input���������d
p

 
� "����������d�
0__inference_sequential_1_layer_call_fn_119089951W7�4
-�*
 �
inputs���������d
p 

 
� "����������d�
0__inference_sequential_1_layer_call_fn_119089960W7�4
-�*
 �
inputs���������d
p

 
� "����������d�
'__inference_signature_wrapper_119089910�O�L
� 
E�B
@
embedding_1_input+�(
embedding_1_input���������d"1�.
,
dense_1!�
dense_1���������d