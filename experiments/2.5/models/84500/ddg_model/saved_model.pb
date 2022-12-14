�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
executor_typestring ��
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
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:�*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:�*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:�*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:�*
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:�@*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
��*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	�@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�c
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�b
value�bB�b B�b
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
'
#_self_saveable_object_factories* 
�

kernel
bias
#_self_saveable_object_factories
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
�

&kernel
'bias
#(_self_saveable_object_factories
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
�
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
�

6kernel
7bias
#8_self_saveable_object_factories
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
�

?kernel
@bias
#A_self_saveable_object_factories
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
'
#H_self_saveable_object_factories* 
�
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
�

Pkernel
Qbias
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*
�

Ykernel
Zbias
#[_self_saveable_object_factories
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*
�

bkernel
cbias
#d_self_saveable_object_factories
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses*
�
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
�
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses* 
�
#y_self_saveable_object_factories
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�serving_default* 
* 
�
0
1
&2
'3
64
75
?6
@7
P8
Q9
Y10
Z11
b12
c13
�14
�15
�16
�17
�18
�19
�20
�21*
�
0
1
&2
'3
64
75
?6
@7
P8
Q9
Y10
Z11
b12
c13
�14
�15
�16
�17
�18
�19
�20
�21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

&0
'1*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

P0
Q1*

P0
Q1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

b0
c1*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������		*
dtype0*$
shape:���������		
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasconv2d_4/kernelconv2d_4/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *0
f+R)
'__inference_signature_wrapper_129287712
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *+
f&R$
"__inference__traced_save_129289046
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasconv2d_4/kernelconv2d_4/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *.
f)R'
%__inference__traced_restore_129289122��
Ԛ
�
D__inference_model_layer_call_and_return_conditional_losses_129287501
inputs_0
inputs_1@
%conv2d_conv2d_readvariableop_resource:�5
&conv2d_biasadd_readvariableop_resource:	�C
'conv2d_1_conv2d_readvariableop_resource:��7
(conv2d_1_biasadd_readvariableop_resource:	�C
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�B
'conv2d_3_conv2d_readvariableop_resource:�@6
(conv2d_3_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@:
&dense_2_matmul_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�:
&dense_3_matmul_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�9
&dense_4_matmul_readvariableop_resource:	�@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�P
conv2d/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{

conv2d/mulMulconv2d/beta:output:0conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�d
conv2d/SigmoidSigmoidconv2d/mul:z:0*
T0*0
_output_shapes
:���������		�{
conv2d/mul_1Mulconv2d/BiasAdd:output:0conv2d/Sigmoid:y:0*
T0*0
_output_shapes
:���������		�h
conv2d/IdentityIdentityconv2d/mul_1:z:0*
T0*0
_output_shapes
:���������		��
conv2d/IdentityN	IdentityNconv2d/mul_1:z:0conv2d/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287353*L
_output_shapes:
8:���������		�:���������		��
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_1/Conv2DConv2Dconv2d/IdentityN:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�R
conv2d_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_1/mulMulconv2d_1/beta:output:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�h
conv2d_1/SigmoidSigmoidconv2d_1/mul:z:0*
T0*0
_output_shapes
:���������		��
conv2d_1/mul_1Mulconv2d_1/BiasAdd:output:0conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:���������		�l
conv2d_1/IdentityIdentityconv2d_1/mul_1:z:0*
T0*0
_output_shapes
:���������		��
conv2d_1/IdentityN	IdentityNconv2d_1/mul_1:z:0conv2d_1/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287367*L
_output_shapes:
8:���������		�:���������		��
max_pooling2d/MaxPoolMaxPoolconv2d_1/IdentityN:output:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������R
conv2d_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_2/mulMulconv2d_2/beta:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:����������h
conv2d_2/SigmoidSigmoidconv2d_2/mul:z:0*
T0*0
_output_shapes
:�����������
conv2d_2/mul_1Mulconv2d_2/BiasAdd:output:0conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:����������l
conv2d_2/IdentityIdentityconv2d_2/mul_1:z:0*
T0*0
_output_shapes
:�����������
conv2d_2/IdentityN	IdentityNconv2d_2/mul_1:z:0conv2d_2/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287382*L
_output_shapes:
8:����������:�����������
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/IdentityN:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@R
conv2d_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_3/mulMulconv2d_3/beta:output:0conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
conv2d_3/SigmoidSigmoidconv2d_3/mul:z:0*
T0*/
_output_shapes
:���������@�
conv2d_3/mul_1Mulconv2d_3/BiasAdd:output:0conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:���������@k
conv2d_3/IdentityIdentityconv2d_3/mul_1:z:0*
T0*/
_output_shapes
:���������@�
conv2d_3/IdentityN	IdentityNconv2d_3/mul_1:z:0conv2d_3/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287396*J
_output_shapes8
6:���������@:���������@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0w
dense/MatMulMatMulinputs_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@O

dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	dense/mulMuldense/beta:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@Y
dense/SigmoidSigmoiddense/mul:z:0*
T0*'
_output_shapes
:���������@o
dense/mul_1Muldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@]
dense/IdentityIdentitydense/mul_1:z:0*
T0*'
_output_shapes
:���������@�
dense/IdentityN	IdentityNdense/mul_1:z:0dense/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287410*:
_output_shapes(
&:���������@:���������@�
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/IdentityN:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� ]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:��������� u
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:��������� a
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:��������� �
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287425*:
_output_shapes(
&:��������� :��������� �
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@R
conv2d_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_4/mulMulconv2d_4/beta:output:0conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
conv2d_4/SigmoidSigmoidconv2d_4/mul:z:0*
T0*/
_output_shapes
:���������@�
conv2d_4/mul_1Mulconv2d_4/BiasAdd:output:0conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:���������@k
conv2d_4/IdentityIdentityconv2d_4/mul_1:z:0*
T0*/
_output_shapes
:���������@�
conv2d_4/IdentityN	IdentityNconv2d_4/mul_1:z:0conv2d_4/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287439*J
_output_shapes8
6:���������@:���������@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeconv2d_4/IdentityN:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    �
flatten_1/ReshapeReshapedense_1/IdentityN:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:��������� Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_2/MatMulMatMulconcatenate/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?v
dense_2/mulMuldense_2/beta:output:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������^
dense_2/SigmoidSigmoiddense_2/mul:z:0*
T0*(
_output_shapes
:����������v
dense_2/mul_1Muldense_2/BiasAdd:output:0dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:����������b
dense_2/IdentityIdentitydense_2/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_2/IdentityN	IdentityNdense_2/mul_1:z:0dense_2/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287459*<
_output_shapes*
(:����������:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_3/MatMulMatMuldense_2/IdentityN:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?v
dense_3/mulMuldense_3/beta:output:0dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������^
dense_3/SigmoidSigmoiddense_3/mul:z:0*
T0*(
_output_shapes
:����������v
dense_3/mul_1Muldense_3/BiasAdd:output:0dense_3/Sigmoid:y:0*
T0*(
_output_shapes
:����������b
dense_3/IdentityIdentitydense_3/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_3/IdentityN	IdentityNdense_3/mul_1:z:0dense_3/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287473*<
_output_shapes*
(:����������:�����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_4/MatMulMatMuldense_3/IdentityN:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Q
dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_4/mulMuldense_4/beta:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������@]
dense_4/SigmoidSigmoiddense_4/mul:z:0*
T0*'
_output_shapes
:���������@u
dense_4/mul_1Muldense_4/BiasAdd:output:0dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:���������@a
dense_4/IdentityIdentitydense_4/mul_1:z:0*
T0*'
_output_shapes
:���������@�
dense_4/IdentityN	IdentityNdense_4/mul_1:z:0dense_4/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287487*:
_output_shapes(
&:���������@:���������@�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_5/MatMulMatMuldense_4/IdentityN:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Y U
/
_output_shapes
:���������		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129287803

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:����������V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:����������f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:����������Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:�����������
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287795*L
_output_shapes:
8:����������:����������l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288321
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
��
�
$__inference__wrapped_model_129286404
input_1
input_2F
+model_conv2d_conv2d_readvariableop_resource:�;
,model_conv2d_biasadd_readvariableop_resource:	�I
-model_conv2d_1_conv2d_readvariableop_resource:��=
.model_conv2d_1_biasadd_readvariableop_resource:	�I
-model_conv2d_2_conv2d_readvariableop_resource:��=
.model_conv2d_2_biasadd_readvariableop_resource:	�H
-model_conv2d_3_conv2d_readvariableop_resource:�@<
.model_conv2d_3_biasadd_readvariableop_resource:@<
*model_dense_matmul_readvariableop_resource:@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@ ;
-model_dense_1_biasadd_readvariableop_resource: G
-model_conv2d_4_conv2d_readvariableop_resource:@@<
.model_conv2d_4_biasadd_readvariableop_resource:@@
,model_dense_2_matmul_readvariableop_resource:
��<
-model_dense_2_biasadd_readvariableop_resource:	�@
,model_dense_3_matmul_readvariableop_resource:
��<
-model_dense_3_biasadd_readvariableop_resource:	�?
,model_dense_4_matmul_readvariableop_resource:	�@;
-model_dense_4_biasadd_readvariableop_resource:@>
,model_dense_5_matmul_readvariableop_resource:@;
-model_dense_5_biasadd_readvariableop_resource:
identity��#model/conv2d/BiasAdd/ReadVariableOp�"model/conv2d/Conv2D/ReadVariableOp�%model/conv2d_1/BiasAdd/ReadVariableOp�$model/conv2d_1/Conv2D/ReadVariableOp�%model/conv2d_2/BiasAdd/ReadVariableOp�$model/conv2d_2/Conv2D/ReadVariableOp�%model/conv2d_3/BiasAdd/ReadVariableOp�$model/conv2d_3/Conv2D/ReadVariableOp�%model/conv2d_4/BiasAdd/ReadVariableOp�$model/conv2d_4/Conv2D/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�$model/dense_4/BiasAdd/ReadVariableOp�#model/dense_4/MatMul/ReadVariableOp�$model/dense_5/BiasAdd/ReadVariableOp�#model/dense_5/MatMul/ReadVariableOp�
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�V
model/conv2d/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/conv2d/mulMulmodel/conv2d/beta:output:0model/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�p
model/conv2d/SigmoidSigmoidmodel/conv2d/mul:z:0*
T0*0
_output_shapes
:���������		��
model/conv2d/mul_1Mulmodel/conv2d/BiasAdd:output:0model/conv2d/Sigmoid:y:0*
T0*0
_output_shapes
:���������		�t
model/conv2d/IdentityIdentitymodel/conv2d/mul_1:z:0*
T0*0
_output_shapes
:���������		��
model/conv2d/IdentityN	IdentityNmodel/conv2d/mul_1:z:0model/conv2d/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286256*L
_output_shapes:
8:���������		�:���������		��
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv2d_1/Conv2DConv2Dmodel/conv2d/IdentityN:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�X
model/conv2d_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/conv2d_1/mulMulmodel/conv2d_1/beta:output:0model/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�t
model/conv2d_1/SigmoidSigmoidmodel/conv2d_1/mul:z:0*
T0*0
_output_shapes
:���������		��
model/conv2d_1/mul_1Mulmodel/conv2d_1/BiasAdd:output:0model/conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:���������		�x
model/conv2d_1/IdentityIdentitymodel/conv2d_1/mul_1:z:0*
T0*0
_output_shapes
:���������		��
model/conv2d_1/IdentityN	IdentityNmodel/conv2d_1/mul_1:z:0model/conv2d_1/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286270*L
_output_shapes:
8:���������		�:���������		��
model/max_pooling2d/MaxPoolMaxPool!model/conv2d_1/IdentityN:output:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv2d_2/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������X
model/conv2d_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/conv2d_2/mulMulmodel/conv2d_2/beta:output:0model/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:����������t
model/conv2d_2/SigmoidSigmoidmodel/conv2d_2/mul:z:0*
T0*0
_output_shapes
:�����������
model/conv2d_2/mul_1Mulmodel/conv2d_2/BiasAdd:output:0model/conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:����������x
model/conv2d_2/IdentityIdentitymodel/conv2d_2/mul_1:z:0*
T0*0
_output_shapes
:�����������
model/conv2d_2/IdentityN	IdentityNmodel/conv2d_2/mul_1:z:0model/conv2d_2/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286285*L
_output_shapes:
8:����������:�����������
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
model/conv2d_3/Conv2DConv2D!model/conv2d_2/IdentityN:output:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
model/conv2d_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/conv2d_3/mulMulmodel/conv2d_3/beta:output:0model/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@s
model/conv2d_3/SigmoidSigmoidmodel/conv2d_3/mul:z:0*
T0*/
_output_shapes
:���������@�
model/conv2d_3/mul_1Mulmodel/conv2d_3/BiasAdd:output:0model/conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:���������@w
model/conv2d_3/IdentityIdentitymodel/conv2d_3/mul_1:z:0*
T0*/
_output_shapes
:���������@�
model/conv2d_3/IdentityN	IdentityNmodel/conv2d_3/mul_1:z:0model/conv2d_3/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286299*J
_output_shapes8
6:���������@:���������@�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense/MatMulMatMulinput_2)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@U
model/dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense/mulMulmodel/dense/beta:output:0model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@e
model/dense/SigmoidSigmoidmodel/dense/mul:z:0*
T0*'
_output_shapes
:���������@�
model/dense/mul_1Mulmodel/dense/BiasAdd:output:0model/dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@i
model/dense/IdentityIdentitymodel/dense/mul_1:z:0*
T0*'
_output_shapes
:���������@�
model/dense/IdentityN	IdentityNmodel/dense/mul_1:z:0model/dense/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286313*:
_output_shapes(
&:���������@:���������@�
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_3/IdentityN:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
model/dense_1/MatMulMatMulmodel/dense/IdentityN:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� W
model/dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense_1/mulMulmodel/dense_1/beta:output:0model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� i
model/dense_1/SigmoidSigmoidmodel/dense_1/mul:z:0*
T0*'
_output_shapes
:��������� �
model/dense_1/mul_1Mulmodel/dense_1/BiasAdd:output:0model/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:��������� m
model/dense_1/IdentityIdentitymodel/dense_1/mul_1:z:0*
T0*'
_output_shapes
:��������� �
model/dense_1/IdentityN	IdentityNmodel/dense_1/mul_1:z:0model/dense_1/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286328*:
_output_shapes(
&:��������� :��������� �
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
model/conv2d_4/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@X
model/conv2d_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/conv2d_4/mulMulmodel/conv2d_4/beta:output:0model/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@s
model/conv2d_4/SigmoidSigmoidmodel/conv2d_4/mul:z:0*
T0*/
_output_shapes
:���������@�
model/conv2d_4/mul_1Mulmodel/conv2d_4/BiasAdd:output:0model/conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:���������@w
model/conv2d_4/IdentityIdentitymodel/conv2d_4/mul_1:z:0*
T0*/
_output_shapes
:���������@�
model/conv2d_4/IdentityN	IdentityNmodel/conv2d_4/mul_1:z:0model/conv2d_4/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286342*J
_output_shapes8
6:���������@:���������@d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshape!model/conv2d_4/IdentityN:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:����������f
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    �
model/flatten_1/ReshapeReshape model/dense_1/IdentityN:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:��������� _
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2model/flatten/Reshape:output:0 model/flatten_1/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_2/MatMulMatMul!model/concatenate/concat:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
model/dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense_2/mulMulmodel/dense_2/beta:output:0model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������j
model/dense_2/SigmoidSigmoidmodel/dense_2/mul:z:0*
T0*(
_output_shapes
:�����������
model/dense_2/mul_1Mulmodel/dense_2/BiasAdd:output:0model/dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:����������n
model/dense_2/IdentityIdentitymodel/dense_2/mul_1:z:0*
T0*(
_output_shapes
:�����������
model/dense_2/IdentityN	IdentityNmodel/dense_2/mul_1:z:0model/dense_2/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286362*<
_output_shapes*
(:����������:�����������
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_3/MatMulMatMul model/dense_2/IdentityN:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
model/dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense_3/mulMulmodel/dense_3/beta:output:0model/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������j
model/dense_3/SigmoidSigmoidmodel/dense_3/mul:z:0*
T0*(
_output_shapes
:�����������
model/dense_3/mul_1Mulmodel/dense_3/BiasAdd:output:0model/dense_3/Sigmoid:y:0*
T0*(
_output_shapes
:����������n
model/dense_3/IdentityIdentitymodel/dense_3/mul_1:z:0*
T0*(
_output_shapes
:�����������
model/dense_3/IdentityN	IdentityNmodel/dense_3/mul_1:z:0model/dense_3/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286376*<
_output_shapes*
(:����������:�����������
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_4/MatMulMatMul model/dense_3/IdentityN:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@W
model/dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/dense_4/mulMulmodel/dense_4/beta:output:0model/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������@i
model/dense_4/SigmoidSigmoidmodel/dense_4/mul:z:0*
T0*'
_output_shapes
:���������@�
model/dense_4/mul_1Mulmodel/dense_4/BiasAdd:output:0model/dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:���������@m
model/dense_4/IdentityIdentitymodel/dense_4/mul_1:z:0*
T0*'
_output_shapes
:���������@�
model/dense_4/IdentityN	IdentityNmodel/dense_4/mul_1:z:0model/dense_4/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286390*:
_output_shapes(
&:���������@:���������@�
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_5/MatMulMatMul model/dense_4/IdentityN:output:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
IdentityIdentitymodel/dense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
}
&__inference_internal_grad_fn_129289005
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
�
&__inference_internal_grad_fn_129288483
result_grads_0
result_grads_1
mul_conv2d_beta
mul_conv2d_biasadd
identity{
mulMulmul_conv2d_betamul_conv2d_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�l
mul_1Mulmul_conv2d_betamul_conv2d_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
O
3__inference_max_pooling2d_1_layer_call_fn_129287835

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129286425�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_dense_3_layer_call_and_return_conditional_losses_129286674

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:�����������
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286666*<
_output_shapes*
(:����������:����������d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�I
�

D__inference_model_layer_call_and_return_conditional_losses_129286721

inputs
inputs_1+
conv2d_129286456:�
conv2d_129286458:	�.
conv2d_1_129286480:��!
conv2d_1_129286482:	�.
conv2d_2_129286505:��!
conv2d_2_129286507:	�-
conv2d_3_129286529:�@ 
conv2d_3_129286531:@!
dense_129286553:@
dense_129286555:@#
dense_1_129286578:@ 
dense_1_129286580: ,
conv2d_4_129286602:@@ 
conv2d_4_129286604:@%
dense_2_129286651:
�� 
dense_2_129286653:	�%
dense_3_129286675:
�� 
dense_3_129286677:	�$
dense_4_129286699:	�@
dense_4_129286701:@#
dense_5_129286715:@
dense_5_129286717:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_129286456conv2d_129286458*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_129286455�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_129286480conv2d_1_129286482*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129286413�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_129286505conv2d_2_129286507*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_129286529conv2d_3_129286531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_129286553dense_129286555*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_129286552�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129286425�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_129286578dense_1_129286580*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_129286577�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_129286602conv2d_4_129286604*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601�
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_129286613�
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_129286621�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_129286630�
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_129286651dense_2_129286653*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_129286650�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_129286675dense_3_129286677*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_129286674�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_129286699dense_4_129286701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_129286698�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_129286715dense_5_129286717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_129286714w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288591
result_grads_0
result_grads_1
mul_conv2d_4_beta
mul_conv2d_4_biasadd
identity~
mulMulmul_conv2d_4_betamul_conv2d_4_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@o
mul_1Mulmul_conv2d_4_betamul_conv2d_4_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
�
&__inference_internal_grad_fn_129288573
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:��������� e
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:��������� J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:��������� T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:��������� Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*N
_input_shapes=
;:��������� :��������� : :��������� :W S
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:��������� 
�
�
E__inference_conv2d_layer_call_and_return_conditional_losses_129287739

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:���������		��
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287731*L
_output_shapes:
8:���������		�:���������		�l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:���������		�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�
�
)__inference_model_layer_call_fn_129286768
input_1
input_2"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�$
	unknown_5:�@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:@

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_129286721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
G
+__inference_flatten_layer_call_fn_129287926

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
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_129286613a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_129287943

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_signature_wrapper_129287712
input_1
input_2"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�$
	unknown_5:�@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:@

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *-
f(R&
$__inference__wrapped_model_129286404o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
Ԛ
�
D__inference_model_layer_call_and_return_conditional_losses_129287660
inputs_0
inputs_1@
%conv2d_conv2d_readvariableop_resource:�5
&conv2d_biasadd_readvariableop_resource:	�C
'conv2d_1_conv2d_readvariableop_resource:��7
(conv2d_1_biasadd_readvariableop_resource:	�C
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�B
'conv2d_3_conv2d_readvariableop_resource:�@6
(conv2d_3_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@:
&dense_2_matmul_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�:
&dense_3_matmul_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�9
&dense_4_matmul_readvariableop_resource:	�@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:
identity��conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�P
conv2d/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{

conv2d/mulMulconv2d/beta:output:0conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�d
conv2d/SigmoidSigmoidconv2d/mul:z:0*
T0*0
_output_shapes
:���������		�{
conv2d/mul_1Mulconv2d/BiasAdd:output:0conv2d/Sigmoid:y:0*
T0*0
_output_shapes
:���������		�h
conv2d/IdentityIdentityconv2d/mul_1:z:0*
T0*0
_output_shapes
:���������		��
conv2d/IdentityN	IdentityNconv2d/mul_1:z:0conv2d/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287512*L
_output_shapes:
8:���������		�:���������		��
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_1/Conv2DConv2Dconv2d/IdentityN:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�R
conv2d_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_1/mulMulconv2d_1/beta:output:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������		�h
conv2d_1/SigmoidSigmoidconv2d_1/mul:z:0*
T0*0
_output_shapes
:���������		��
conv2d_1/mul_1Mulconv2d_1/BiasAdd:output:0conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:���������		�l
conv2d_1/IdentityIdentityconv2d_1/mul_1:z:0*
T0*0
_output_shapes
:���������		��
conv2d_1/IdentityN	IdentityNconv2d_1/mul_1:z:0conv2d_1/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287526*L
_output_shapes:
8:���������		�:���������		��
max_pooling2d/MaxPoolMaxPoolconv2d_1/IdentityN:output:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������R
conv2d_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_2/mulMulconv2d_2/beta:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:����������h
conv2d_2/SigmoidSigmoidconv2d_2/mul:z:0*
T0*0
_output_shapes
:�����������
conv2d_2/mul_1Mulconv2d_2/BiasAdd:output:0conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:����������l
conv2d_2/IdentityIdentityconv2d_2/mul_1:z:0*
T0*0
_output_shapes
:�����������
conv2d_2/IdentityN	IdentityNconv2d_2/mul_1:z:0conv2d_2/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287541*L
_output_shapes:
8:����������:�����������
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/IdentityN:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@R
conv2d_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_3/mulMulconv2d_3/beta:output:0conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
conv2d_3/SigmoidSigmoidconv2d_3/mul:z:0*
T0*/
_output_shapes
:���������@�
conv2d_3/mul_1Mulconv2d_3/BiasAdd:output:0conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:���������@k
conv2d_3/IdentityIdentityconv2d_3/mul_1:z:0*
T0*/
_output_shapes
:���������@�
conv2d_3/IdentityN	IdentityNconv2d_3/mul_1:z:0conv2d_3/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287555*J
_output_shapes8
6:���������@:���������@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0w
dense/MatMulMatMulinputs_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@O

dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	dense/mulMuldense/beta:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@Y
dense/SigmoidSigmoiddense/mul:z:0*
T0*'
_output_shapes
:���������@o
dense/mul_1Muldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@]
dense/IdentityIdentitydense/mul_1:z:0*
T0*'
_output_shapes
:���������@�
dense/IdentityN	IdentityNdense/mul_1:z:0dense/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287569*:
_output_shapes(
&:���������@:���������@�
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/IdentityN:output:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� ]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:��������� u
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:��������� a
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:��������� �
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287584*:
_output_shapes(
&:��������� :��������� �
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@R
conv2d_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
conv2d_4/mulMulconv2d_4/beta:output:0conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
conv2d_4/SigmoidSigmoidconv2d_4/mul:z:0*
T0*/
_output_shapes
:���������@�
conv2d_4/mul_1Mulconv2d_4/BiasAdd:output:0conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:���������@k
conv2d_4/IdentityIdentityconv2d_4/mul_1:z:0*
T0*/
_output_shapes
:���������@�
conv2d_4/IdentityN	IdentityNconv2d_4/mul_1:z:0conv2d_4/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287598*J
_output_shapes8
6:���������@:���������@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeconv2d_4/IdentityN:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    �
flatten_1/ReshapeReshapedense_1/IdentityN:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:��������� Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_2/MatMulMatMulconcatenate/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?v
dense_2/mulMuldense_2/beta:output:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������^
dense_2/SigmoidSigmoiddense_2/mul:z:0*
T0*(
_output_shapes
:����������v
dense_2/mul_1Muldense_2/BiasAdd:output:0dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:����������b
dense_2/IdentityIdentitydense_2/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_2/IdentityN	IdentityNdense_2/mul_1:z:0dense_2/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287618*<
_output_shapes*
(:����������:�����������
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_3/MatMulMatMuldense_2/IdentityN:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?v
dense_3/mulMuldense_3/beta:output:0dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������^
dense_3/SigmoidSigmoiddense_3/mul:z:0*
T0*(
_output_shapes
:����������v
dense_3/mul_1Muldense_3/BiasAdd:output:0dense_3/Sigmoid:y:0*
T0*(
_output_shapes
:����������b
dense_3/IdentityIdentitydense_3/mul_1:z:0*
T0*(
_output_shapes
:�����������
dense_3/IdentityN	IdentityNdense_3/mul_1:z:0dense_3/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287632*<
_output_shapes*
(:����������:�����������
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_4/MatMulMatMuldense_3/IdentityN:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Q
dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
dense_4/mulMuldense_4/beta:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������@]
dense_4/SigmoidSigmoiddense_4/mul:z:0*
T0*'
_output_shapes
:���������@u
dense_4/mul_1Muldense_4/BiasAdd:output:0dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:���������@a
dense_4/IdentityIdentitydense_4/mul_1:z:0*
T0*'
_output_shapes
:���������@�
dense_4/IdentityN	IdentityNdense_4/mul_1:z:0dense_4/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287646*:
_output_shapes(
&:���������@:���������@�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_5/MatMulMatMuldense_4/IdentityN:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Y U
/
_output_shapes
:���������		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�W
�
%__inference__traced_restore_129289122
file_prefix9
assignvariableop_conv2d_kernel:�-
assignvariableop_1_conv2d_bias:	�>
"assignvariableop_2_conv2d_1_kernel:��/
 assignvariableop_3_conv2d_1_bias:	�>
"assignvariableop_4_conv2d_2_kernel:��/
 assignvariableop_5_conv2d_2_bias:	�=
"assignvariableop_6_conv2d_3_kernel:�@.
 assignvariableop_7_conv2d_3_bias:@1
assignvariableop_8_dense_kernel:@+
assignvariableop_9_dense_bias:@=
#assignvariableop_10_conv2d_4_kernel:@@/
!assignvariableop_11_conv2d_4_bias:@4
"assignvariableop_12_dense_1_kernel:@ .
 assignvariableop_13_dense_1_bias: 6
"assignvariableop_14_dense_2_kernel:
��/
 assignvariableop_15_dense_2_bias:	�6
"assignvariableop_16_dense_3_kernel:
��/
 assignvariableop_17_dense_3_bias:	�5
"assignvariableop_18_dense_4_kernel:	�@.
 assignvariableop_19_dense_4_bias:@4
"assignvariableop_20_dense_5_kernel:@.
 assignvariableop_21_dense_5_bias:
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_5_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
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
�
�
+__inference_dense_5_layer_call_fn_129288046

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_129286714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288501
result_grads_0
result_grads_1
mul_conv2d_1_beta
mul_conv2d_1_biasadd
identity
mulMulmul_conv2d_1_betamul_conv2d_1_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�p
mul_1Mulmul_conv2d_1_betamul_conv2d_1_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
�
&__inference_internal_grad_fn_129288231
result_grads_0
result_grads_1
mul_model_conv2d_4_beta
mul_model_conv2d_4_biasadd
identity�
mulMulmul_model_conv2d_4_betamul_model_conv2d_4_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@{
mul_1Mulmul_model_conv2d_4_betamul_model_conv2d_4_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
�
&__inference_internal_grad_fn_129288681
result_grads_0
result_grads_1
mul_conv2d_1_beta
mul_conv2d_1_biasadd
identity
mulMulmul_conv2d_1_betamul_conv2d_1_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�p
mul_1Mulmul_conv2d_1_betamul_conv2d_1_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_129287932

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_dense_3_layer_call_fn_129287992

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_129286674p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�2
�
"__inference__traced_save_129289046
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:��:�:��:�:�@:@:@:@:@@:@:@ : :
��:�:
��:�:	�@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-)
'
_output_shapes
:�@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
�
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_129286621

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
t
J__inference_concatenate_layer_call_and_return_conditional_losses_129286630

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:��������� :P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288285
result_grads_0
result_grads_1
mul_model_dense_4_beta
mul_model_dense_4_biasadd
identity�
mulMulmul_model_dense_4_betamul_model_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@q
mul_1Mulmul_model_dense_4_betamul_model_dense_4_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
�
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129287766

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:���������		��
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287758*L
_output_shapes:
8:���������		�:���������		�l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:���������		�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������		�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
+__inference_dense_1_layer_call_fn_129287903

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_129286577o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129286413

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288267
result_grads_0
result_grads_1
mul_model_dense_3_beta
mul_model_dense_3_biasadd
identity�
mulMulmul_model_dense_3_betamul_model_dense_3_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������r
mul_1Mulmul_model_dense_3_betamul_model_dense_3_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
�
+__inference_dense_2_layer_call_fn_129287965

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_129286650p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288969
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������V
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
�
&__inference_internal_grad_fn_129288807
result_grads_0
result_grads_1
mul_dense_3_beta
mul_dense_3_biasadd
identityu
mulMulmul_dense_3_betamul_dense_3_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������f
mul_1Mulmul_dense_3_betamul_dense_3_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
�
F__inference_dense_4_layer_call_and_return_conditional_losses_129288037

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129288029*:
_output_shapes(
&:���������@:���������@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_conv2d_1_layer_call_fn_129287748

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������		�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������		�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288519
result_grads_0
result_grads_1
mul_conv2d_2_beta
mul_conv2d_2_biasadd
identity
mulMulmul_conv2d_2_betamul_conv2d_2_biasadd^result_grads_0*
T0*0
_output_shapes
:����������V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:����������p
mul_1Mulmul_conv2d_2_betamul_conv2d_2_biasadd*
T0*0
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:����������[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:����������]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:����������b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:����������Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*i
_input_shapesX
V:����������:����������: :����������:` \
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:����������
�
�
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286593*J
_output_shapes8
6:���������@:���������@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288537
result_grads_0
result_grads_1
mul_conv2d_3_beta
mul_conv2d_3_biasadd
identity~
mulMulmul_conv2d_3_betamul_conv2d_3_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@o
mul_1Mulmul_conv2d_3_betamul_conv2d_3_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
}
&__inference_internal_grad_fn_129288375
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
}
&__inference_internal_grad_fn_129288879
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:����������V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:����������^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:����������[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:����������]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:����������b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:����������Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*i
_input_shapesX
V:����������:����������: :����������:` \
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:����������
�
�
F__inference_dense_1_layer_call_and_return_conditional_losses_129286577

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:��������� ]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:��������� �
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286569*:
_output_shapes(
&:��������� :��������� c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288861
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
�
&__inference_internal_grad_fn_129288771
result_grads_0
result_grads_1
mul_conv2d_4_beta
mul_conv2d_4_biasadd
identity~
mulMulmul_conv2d_4_betamul_conv2d_4_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@o
mul_1Mulmul_conv2d_4_betamul_conv2d_4_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
�
&__inference_internal_grad_fn_129288249
result_grads_0
result_grads_1
mul_model_dense_2_beta
mul_model_dense_2_biasadd
identity�
mulMulmul_model_dense_2_betamul_model_dense_2_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������r
mul_1Mulmul_model_dense_2_betamul_model_dense_2_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
�
&__inference_internal_grad_fn_129288177
result_grads_0
result_grads_1
mul_model_conv2d_3_beta
mul_model_conv2d_3_biasadd
identity�
mulMulmul_model_conv2d_3_betamul_model_conv2d_3_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@{
mul_1Mulmul_model_conv2d_3_betamul_model_conv2d_3_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
�
F__inference_dense_3_layer_call_and_return_conditional_losses_129288010

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:�����������
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129288002*<
_output_shapes*
(:����������:����������d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288303
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�	
�
F__inference_dense_5_layer_call_and_return_conditional_losses_129288056

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288393
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:��������� U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:��������� J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:��������� T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:��������� Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*N
_input_shapes=
;:��������� :��������� : :��������� :W S
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:��������� 
�
�
+__inference_dense_4_layer_call_fn_129288019

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_129286698o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288933
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
}
&__inference_internal_grad_fn_129288339
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:����������V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:����������^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:����������[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:����������]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:����������b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:����������Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*i
_input_shapesX
V:����������:����������: :����������:` \
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:����������
�
�
&__inference_internal_grad_fn_129288789
result_grads_0
result_grads_1
mul_dense_2_beta
mul_dense_2_biasadd
identityu
mulMulmul_dense_2_betamul_dense_2_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������f
mul_1Mulmul_dense_2_betamul_dense_2_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
}
&__inference_internal_grad_fn_129288357
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
M
1__inference_max_pooling2d_layer_call_fn_129287771

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129286413�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288897
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
�
&__inference_internal_grad_fn_129288663
result_grads_0
result_grads_1
mul_conv2d_beta
mul_conv2d_biasadd
identity{
mulMulmul_conv2d_betamul_conv2d_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�l
mul_1Mulmul_conv2d_betamul_conv2d_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
}
&__inference_internal_grad_fn_129288429
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������V
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
�
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:����������V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:����������f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:����������Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:�����������
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286496*L
_output_shapes:
8:����������:����������l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286520*J
_output_shapes8
6:���������@:���������@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_layer_call_and_return_conditional_losses_129287956
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:��������� :R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
}
&__inference_internal_grad_fn_129288843
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
�
*__inference_conv2d_layer_call_fn_129287721

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_129286455x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������		�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288987
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������V
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
j
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129287840

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�I
�

D__inference_model_layer_call_and_return_conditional_losses_129287177
input_1
input_2+
conv2d_129287116:�
conv2d_129287118:	�.
conv2d_1_129287121:��!
conv2d_1_129287123:	�.
conv2d_2_129287127:��!
conv2d_2_129287129:	�-
conv2d_3_129287132:�@ 
conv2d_3_129287134:@!
dense_129287137:@
dense_129287139:@#
dense_1_129287143:@ 
dense_1_129287145: ,
conv2d_4_129287148:@@ 
conv2d_4_129287150:@%
dense_2_129287156:
�� 
dense_2_129287158:	�%
dense_3_129287161:
�� 
dense_3_129287163:	�$
dense_4_129287166:	�@
dense_4_129287168:@#
dense_5_129287171:@
dense_5_129287173:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_129287116conv2d_129287118*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_129286455�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_129287121conv2d_1_129287123*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129286413�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_129287127conv2d_2_129287129*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_129287132conv2d_3_129287134*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_129287137dense_129287139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_129286552�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129286425�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_129287143dense_1_129287145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_129286577�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_129287148conv2d_4_129287150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601�
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_129286613�
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_129286621�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_129286630�
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_129287156dense_2_129287158*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_129286650�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_129287161dense_3_129287163*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_129286674�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_129287166dense_4_129287168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_129286698�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_129287171dense_5_129287173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_129286714w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
j
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129286425

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129287776

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�I
�

D__inference_model_layer_call_and_return_conditional_losses_129287015

inputs
inputs_1+
conv2d_129286954:�
conv2d_129286956:	�.
conv2d_1_129286959:��!
conv2d_1_129286961:	�.
conv2d_2_129286965:��!
conv2d_2_129286967:	�-
conv2d_3_129286970:�@ 
conv2d_3_129286972:@!
dense_129286975:@
dense_129286977:@#
dense_1_129286981:@ 
dense_1_129286983: ,
conv2d_4_129286986:@@ 
conv2d_4_129286988:@%
dense_2_129286994:
�� 
dense_2_129286996:	�%
dense_3_129286999:
�� 
dense_3_129287001:	�$
dense_4_129287004:	�@
dense_4_129287006:@#
dense_5_129287009:@
dense_5_129287011:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_129286954conv2d_129286956*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_129286455�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_129286959conv2d_1_129286961*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129286413�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_129286965conv2d_2_129286967*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_129286970conv2d_3_129286972*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_129286975dense_129286977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_129286552�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129286425�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_129286981dense_1_129286983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_129286577�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_129286986conv2d_4_129286988*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601�
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_129286613�
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_129286621�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_129286630�
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_129286994dense_2_129286996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_129286650�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_129286999dense_3_129287001*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_129286674�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_129287004dense_4_129287006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_129286698�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_129287009dense_5_129287011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_129286714w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288159
result_grads_0
result_grads_1
mul_model_conv2d_2_beta
mul_model_conv2d_2_biasadd
identity�
mulMulmul_model_conv2d_2_betamul_model_conv2d_2_biasadd^result_grads_0*
T0*0
_output_shapes
:����������V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:����������|
mul_1Mulmul_model_conv2d_2_betamul_model_conv2d_2_biasadd*
T0*0
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:����������[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:����������]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:����������b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:����������Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*i
_input_shapesX
V:����������:����������: :����������:` \
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:����������
�
�
&__inference_internal_grad_fn_129288141
result_grads_0
result_grads_1
mul_model_conv2d_1_beta
mul_model_conv2d_1_biasadd
identity�
mulMulmul_model_conv2d_1_betamul_model_conv2d_1_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�|
mul_1Mulmul_model_conv2d_1_betamul_model_conv2d_1_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
b
F__inference_flatten_layer_call_and_return_conditional_losses_129286613

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129287830

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287822*J
_output_shapes8
6:���������@:���������@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288123
result_grads_0
result_grads_1
mul_model_conv2d_beta
mul_model_conv2d_biasadd
identity�
mulMulmul_model_conv2d_betamul_model_conv2d_biasadd^result_grads_0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�x
mul_1Mulmul_model_conv2d_betamul_model_conv2d_biasadd*
T0*0
_output_shapes
:���������		�J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:���������		�J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:���������		�]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:���������		�b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:���������		�"
identityIdentity:output:0*i
_input_shapesX
V:���������		�:���������		�: :���������		�:` \
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:���������		�
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:���������		�
�
}
&__inference_internal_grad_fn_129288447
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������V
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
�
,__inference_conv2d_3_layer_call_fn_129287812

inputs"
unknown:�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_5_layer_call_and_return_conditional_losses_129286714

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288951
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:��������� U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:��������� J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:��������� T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:��������� Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*N
_input_shapes=
;:��������� :��������� : :��������� :W S
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:��������� 
�
�
)__inference_model_layer_call_fn_129287112
input_1
input_2"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�$
	unknown_5:�@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:@

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_129287015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
}
&__inference_internal_grad_fn_129288915
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
�
E__inference_conv2d_layer_call_and_return_conditional_losses_129286455

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:���������		��
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286447*L
_output_shapes:
8:���������		�:���������		�l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:���������		�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������		
 
_user_specified_nameinputs
�
�
D__inference_dense_layer_call_and_return_conditional_losses_129287867

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287859*:
_output_shapes(
&:���������@:���������@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288699
result_grads_0
result_grads_1
mul_conv2d_2_beta
mul_conv2d_2_biasadd
identity
mulMulmul_conv2d_2_betamul_conv2d_2_biasadd^result_grads_0*
T0*0
_output_shapes
:����������V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:����������p
mul_1Mulmul_conv2d_2_betamul_conv2d_2_biasadd*
T0*0
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:����������[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:����������]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:����������b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:����������Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*i
_input_shapesX
V:����������:����������: :����������:` \
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:����������
�
�
&__inference_internal_grad_fn_129288645
result_grads_0
result_grads_1
mul_dense_4_beta
mul_dense_4_biasadd
identityt
mulMulmul_dense_4_betamul_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@e
mul_1Mulmul_dense_4_betamul_dense_4_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
�
)__inference_dense_layer_call_fn_129287849

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_129286552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129287894

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287886*J
_output_shapes8
6:���������@:���������@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288753
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:��������� e
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:��������� J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:��������� T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:��������� Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*N
_input_shapes=
;:��������� :��������� : :��������� :W S
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:��������� 
�
�
,__inference_conv2d_2_layer_call_fn_129287785

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288735
result_grads_0
result_grads_1
mul_dense_beta
mul_dense_biasadd
identityp
mulMulmul_dense_betamul_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@a
mul_1Mulmul_dense_betamul_dense_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
�
&__inference_internal_grad_fn_129288195
result_grads_0
result_grads_1
mul_model_dense_beta
mul_model_dense_biasadd
identity|
mulMulmul_model_dense_betamul_model_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@m
mul_1Mulmul_model_dense_betamul_model_dense_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
[
/__inference_concatenate_layer_call_fn_129287949
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_129286630a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:��������� :R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
�
)__inference_model_layer_call_fn_129287342
inputs_0
inputs_1"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�$
	unknown_5:�@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:@

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_129287015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
F__inference_dense_1_layer_call_and_return_conditional_losses_129287921

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:��������� ]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:��������� �
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287913*:
_output_shapes(
&:��������� :��������� c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_dense_layer_call_and_return_conditional_losses_129286552

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286544*:
_output_shapes(
&:���������@:���������@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_dense_2_layer_call_and_return_conditional_losses_129286650

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:�����������
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286642*<
_output_shapes*
(:����������:����������d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_dense_4_layer_call_and_return_conditional_losses_129286698

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:���������@�
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286690*:
_output_shapes(
&:���������@:���������@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288213
result_grads_0
result_grads_1
mul_model_dense_1_beta
mul_model_dense_1_biasadd
identity�
mulMulmul_model_dense_1_betamul_model_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:��������� M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:��������� q
mul_1Mulmul_model_dense_1_betamul_model_dense_1_biasadd*
T0*'
_output_shapes
:��������� J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:��������� R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:��������� J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:��������� T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:��������� Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:��������� Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*N
_input_shapes=
;:��������� :��������� : :��������� :W S
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:��������� 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:��������� 
�
�
&__inference_internal_grad_fn_129288825
result_grads_0
result_grads_1
mul_dense_4_beta
mul_dense_4_biasadd
identityt
mulMulmul_dense_4_betamul_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@e
mul_1Mulmul_dense_4_betamul_dense_4_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
�
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������		�I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:���������		�V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:���������		�f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:���������		�Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:���������		��
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129286471*L
_output_shapes:
8:���������		�:���������		�l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:���������		�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������		�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������		�
 
_user_specified_nameinputs
�
�
F__inference_dense_2_layer_call_and_return_conditional_losses_129287983

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  �?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:�����������
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-129287975*<
_output_shapes*
(:����������:����������d

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_internal_grad_fn_129288609
result_grads_0
result_grads_1
mul_dense_2_beta
mul_dense_2_biasadd
identityu
mulMulmul_dense_2_betamul_dense_2_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������f
mul_1Mulmul_dense_2_betamul_dense_2_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�I
�

D__inference_model_layer_call_and_return_conditional_losses_129287242
input_1
input_2+
conv2d_129287181:�
conv2d_129287183:	�.
conv2d_1_129287186:��!
conv2d_1_129287188:	�.
conv2d_2_129287192:��!
conv2d_2_129287194:	�-
conv2d_3_129287197:�@ 
conv2d_3_129287199:@!
dense_129287202:@
dense_129287204:@#
dense_1_129287208:@ 
dense_1_129287210: ,
conv2d_4_129287213:@@ 
conv2d_4_129287215:@%
dense_2_129287221:
�� 
dense_2_129287223:	�%
dense_3_129287226:
�� 
dense_3_129287228:	�$
dense_4_129287231:	�@
dense_4_129287233:@#
dense_5_129287236:@
dense_5_129287238:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall� conv2d_4/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_129287181conv2d_129287183*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *N
fIRG
E__inference_conv2d_layer_call_and_return_conditional_losses_129286455�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_129287186conv2d_1_129287188*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������		�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129286413�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_129287192conv2d_2_129287194*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_129287197conv2d_3_129287199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_129287202dense_129287204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_dense_layer_call_and_return_conditional_losses_129286552�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129286425�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_129287208dense_1_129287210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_1_layer_call_and_return_conditional_losses_129286577�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_129287213conv2d_4_129287215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601�
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_129286613�
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_129286621�
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_129286630�
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_129287221dense_2_129287223*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_2_layer_call_and_return_conditional_losses_129286650�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_129287226dense_3_129287228*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_3_layer_call_and_return_conditional_losses_129286674�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_129287231dense_4_129287233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_4_layer_call_and_return_conditional_losses_129286698�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_129287236dense_5_129287238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *O
fJRH
F__inference_dense_5_layer_call_and_return_conditional_losses_129286714w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������		
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
&__inference_internal_grad_fn_129288717
result_grads_0
result_grads_1
mul_conv2d_3_beta
mul_conv2d_3_biasadd
identity~
mulMulmul_conv2d_3_betamul_conv2d_3_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@o
mul_1Mulmul_conv2d_3_betamul_conv2d_3_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
}
&__inference_internal_grad_fn_129288411
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:���������@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:���������@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:���������@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:���������@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:���������@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:���������@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:���������@"
identityIdentity:output:0*f
_input_shapesU
S:���������@:���������@: :���������@:_ [
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:���������@
�
�
&__inference_internal_grad_fn_129288627
result_grads_0
result_grads_1
mul_dense_3_beta
mul_dense_3_biasadd
identityu
mulMulmul_dense_3_betamul_dense_3_biasadd^result_grads_0*
T0*(
_output_shapes
:����������N
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:����������f
mul_1Mulmul_dense_3_betamul_dense_3_biasadd*
T0*(
_output_shapes
:����������J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:����������S
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:����������J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:����������U
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:����������Z
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:����������R
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*Q
_input_shapes@
>:����������:����������: :����������:X T
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:����������
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:����������
�
�
&__inference_internal_grad_fn_129288555
result_grads_0
result_grads_1
mul_dense_beta
mul_dense_biasadd
identityp
mulMulmul_dense_betamul_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@a
mul_1Mulmul_dense_betamul_dense_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
I
-__inference_flatten_1_layer_call_fn_129287937

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_129286621`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
}
&__inference_internal_grad_fn_129288465
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:���������@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:���������@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:���������@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:���������@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:���������@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*N
_input_shapes=
;:���������@:���������@: :���������@:W S
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:���������@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:���������@
�
�
,__inference_conv2d_4_layer_call_fn_129287876

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_model_layer_call_fn_129287292
inputs_0
inputs_1"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�$
	unknown_5:�@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:	�@

unknown_18:@

unknown_19:@

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_129286721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:���������		:���������: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1B
&__inference_internal_grad_fn_129288123CustomGradient-129286256B
&__inference_internal_grad_fn_129288141CustomGradient-129286270B
&__inference_internal_grad_fn_129288159CustomGradient-129286285B
&__inference_internal_grad_fn_129288177CustomGradient-129286299B
&__inference_internal_grad_fn_129288195CustomGradient-129286313B
&__inference_internal_grad_fn_129288213CustomGradient-129286328B
&__inference_internal_grad_fn_129288231CustomGradient-129286342B
&__inference_internal_grad_fn_129288249CustomGradient-129286362B
&__inference_internal_grad_fn_129288267CustomGradient-129286376B
&__inference_internal_grad_fn_129288285CustomGradient-129286390B
&__inference_internal_grad_fn_129288303CustomGradient-129286447B
&__inference_internal_grad_fn_129288321CustomGradient-129286471B
&__inference_internal_grad_fn_129288339CustomGradient-129286496B
&__inference_internal_grad_fn_129288357CustomGradient-129286520B
&__inference_internal_grad_fn_129288375CustomGradient-129286544B
&__inference_internal_grad_fn_129288393CustomGradient-129286569B
&__inference_internal_grad_fn_129288411CustomGradient-129286593B
&__inference_internal_grad_fn_129288429CustomGradient-129286642B
&__inference_internal_grad_fn_129288447CustomGradient-129286666B
&__inference_internal_grad_fn_129288465CustomGradient-129286690B
&__inference_internal_grad_fn_129288483CustomGradient-129287353B
&__inference_internal_grad_fn_129288501CustomGradient-129287367B
&__inference_internal_grad_fn_129288519CustomGradient-129287382B
&__inference_internal_grad_fn_129288537CustomGradient-129287396B
&__inference_internal_grad_fn_129288555CustomGradient-129287410B
&__inference_internal_grad_fn_129288573CustomGradient-129287425B
&__inference_internal_grad_fn_129288591CustomGradient-129287439B
&__inference_internal_grad_fn_129288609CustomGradient-129287459B
&__inference_internal_grad_fn_129288627CustomGradient-129287473B
&__inference_internal_grad_fn_129288645CustomGradient-129287487B
&__inference_internal_grad_fn_129288663CustomGradient-129287512B
&__inference_internal_grad_fn_129288681CustomGradient-129287526B
&__inference_internal_grad_fn_129288699CustomGradient-129287541B
&__inference_internal_grad_fn_129288717CustomGradient-129287555B
&__inference_internal_grad_fn_129288735CustomGradient-129287569B
&__inference_internal_grad_fn_129288753CustomGradient-129287584B
&__inference_internal_grad_fn_129288771CustomGradient-129287598B
&__inference_internal_grad_fn_129288789CustomGradient-129287618B
&__inference_internal_grad_fn_129288807CustomGradient-129287632B
&__inference_internal_grad_fn_129288825CustomGradient-129287646B
&__inference_internal_grad_fn_129288843CustomGradient-129287731B
&__inference_internal_grad_fn_129288861CustomGradient-129287758B
&__inference_internal_grad_fn_129288879CustomGradient-129287795B
&__inference_internal_grad_fn_129288897CustomGradient-129287822B
&__inference_internal_grad_fn_129288915CustomGradient-129287859B
&__inference_internal_grad_fn_129288933CustomGradient-129287886B
&__inference_internal_grad_fn_129288951CustomGradient-129287913B
&__inference_internal_grad_fn_129288969CustomGradient-129287975B
&__inference_internal_grad_fn_129288987CustomGradient-129288002B
&__inference_internal_grad_fn_129289005CustomGradient-129288029"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������		
;
input_20
serving_default_input_2:0���������;
dense_50
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer_with_weights-9
layer-16
layer_with_weights-10
layer-17

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
�

kernel
bias
#_self_saveable_object_factories
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
�

&kernel
'bias
#(_self_saveable_object_factories
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
�

6kernel
7bias
#8_self_saveable_object_factories
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
�

?kernel
@bias
#A_self_saveable_object_factories
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
D
#H_self_saveable_object_factories"
_tf_keras_input_layer
�
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Pkernel
Qbias
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ykernel
Zbias
#[_self_saveable_object_factories
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
�

bkernel
cbias
#d_self_saveable_object_factories
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#y_self_saveable_object_factories
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
$�_self_saveable_object_factories
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
�
0
1
&2
'3
64
75
?6
@7
P8
Q9
Y10
Z11
b12
c13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
�
0
1
&2
'3
64
75
?6
@7
P8
Q9
Y10
Z11
b12
c13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_model_layer_call_fn_129286768
)__inference_model_layer_call_fn_129287292
)__inference_model_layer_call_fn_129287342
)__inference_model_layer_call_fn_129287112�
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
D__inference_model_layer_call_and_return_conditional_losses_129287501
D__inference_model_layer_call_and_return_conditional_losses_129287660
D__inference_model_layer_call_and_return_conditional_losses_129287177
D__inference_model_layer_call_and_return_conditional_losses_129287242�
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
�B�
$__inference__wrapped_model_129286404input_1input_2"�
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
 "
trackable_dict_wrapper
(:&�2conv2d/kernel
:�2conv2d/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_conv2d_layer_call_fn_129287721�
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
E__inference_conv2d_layer_call_and_return_conditional_losses_129287739�
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
+:)��2conv2d_1/kernel
:�2conv2d_1/bias
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_conv2d_1_layer_call_fn_129287748�
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
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129287766�
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_max_pooling2d_layer_call_fn_129287771�
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
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129287776�
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
+:)��2conv2d_2/kernel
:�2conv2d_2/bias
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_conv2d_2_layer_call_fn_129287785�
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
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129287803�
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
*:(�@2conv2d_3/kernel
:@2conv2d_3/bias
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_conv2d_3_layer_call_fn_129287812�
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
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129287830�
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
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�2�
3__inference_max_pooling2d_1_layer_call_fn_129287835�
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
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129287840�
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
:@2dense/kernel
:@2
dense/bias
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_dense_layer_call_fn_129287849�
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
D__inference_dense_layer_call_and_return_conditional_losses_129287867�
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
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_conv2d_4_layer_call_fn_129287876�
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
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129287894�
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
 :@ 2dense_1/kernel
: 2dense_1/bias
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_1_layer_call_fn_129287903�
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
F__inference_dense_1_layer_call_and_return_conditional_losses_129287921�
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_flatten_layer_call_fn_129287926�
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
F__inference_flatten_layer_call_and_return_conditional_losses_129287932�
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_flatten_1_layer_call_fn_129287937�
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
H__inference_flatten_1_layer_call_and_return_conditional_losses_129287943�
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_concatenate_layer_call_fn_129287949�
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
J__inference_concatenate_layer_call_and_return_conditional_losses_129287956�
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
": 
��2dense_2/kernel
:�2dense_2/bias
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_2_layer_call_fn_129287965�
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
F__inference_dense_2_layer_call_and_return_conditional_losses_129287983�
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
": 
��2dense_3/kernel
:�2dense_3/bias
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_3_layer_call_fn_129287992�
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
F__inference_dense_3_layer_call_and_return_conditional_losses_129288010�
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
!:	�@2dense_4/kernel
:@2dense_4/bias
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_4_layer_call_fn_129288019�
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
F__inference_dense_4_layer_call_and_return_conditional_losses_129288037�
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
 :@2dense_5/kernel
:2dense_5/bias
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_5_layer_call_fn_129288046�
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
F__inference_dense_5_layer_call_and_return_conditional_losses_129288056�
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
'__inference_signature_wrapper_129287712input_1input_2"�
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
 
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
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
=b;
model/conv2d/beta:0$__inference__wrapped_model_129286404
@b>
model/conv2d/BiasAdd:0$__inference__wrapped_model_129286404
?b=
model/conv2d_1/beta:0$__inference__wrapped_model_129286404
Bb@
model/conv2d_1/BiasAdd:0$__inference__wrapped_model_129286404
?b=
model/conv2d_2/beta:0$__inference__wrapped_model_129286404
Bb@
model/conv2d_2/BiasAdd:0$__inference__wrapped_model_129286404
?b=
model/conv2d_3/beta:0$__inference__wrapped_model_129286404
Bb@
model/conv2d_3/BiasAdd:0$__inference__wrapped_model_129286404
<b:
model/dense/beta:0$__inference__wrapped_model_129286404
?b=
model/dense/BiasAdd:0$__inference__wrapped_model_129286404
>b<
model/dense_1/beta:0$__inference__wrapped_model_129286404
Ab?
model/dense_1/BiasAdd:0$__inference__wrapped_model_129286404
?b=
model/conv2d_4/beta:0$__inference__wrapped_model_129286404
Bb@
model/conv2d_4/BiasAdd:0$__inference__wrapped_model_129286404
>b<
model/dense_2/beta:0$__inference__wrapped_model_129286404
Ab?
model/dense_2/BiasAdd:0$__inference__wrapped_model_129286404
>b<
model/dense_3/beta:0$__inference__wrapped_model_129286404
Ab?
model/dense_3/BiasAdd:0$__inference__wrapped_model_129286404
>b<
model/dense_4/beta:0$__inference__wrapped_model_129286404
Ab?
model/dense_4/BiasAdd:0$__inference__wrapped_model_129286404
QbO
beta:0E__inference_conv2d_layer_call_and_return_conditional_losses_129286455
TbR
	BiasAdd:0E__inference_conv2d_layer_call_and_return_conditional_losses_129286455
SbQ
beta:0G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479
VbT
	BiasAdd:0G__inference_conv2d_1_layer_call_and_return_conditional_losses_129286479
SbQ
beta:0G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504
VbT
	BiasAdd:0G__inference_conv2d_2_layer_call_and_return_conditional_losses_129286504
SbQ
beta:0G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528
VbT
	BiasAdd:0G__inference_conv2d_3_layer_call_and_return_conditional_losses_129286528
PbN
beta:0D__inference_dense_layer_call_and_return_conditional_losses_129286552
SbQ
	BiasAdd:0D__inference_dense_layer_call_and_return_conditional_losses_129286552
RbP
beta:0F__inference_dense_1_layer_call_and_return_conditional_losses_129286577
UbS
	BiasAdd:0F__inference_dense_1_layer_call_and_return_conditional_losses_129286577
SbQ
beta:0G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601
VbT
	BiasAdd:0G__inference_conv2d_4_layer_call_and_return_conditional_losses_129286601
RbP
beta:0F__inference_dense_2_layer_call_and_return_conditional_losses_129286650
UbS
	BiasAdd:0F__inference_dense_2_layer_call_and_return_conditional_losses_129286650
RbP
beta:0F__inference_dense_3_layer_call_and_return_conditional_losses_129286674
UbS
	BiasAdd:0F__inference_dense_3_layer_call_and_return_conditional_losses_129286674
RbP
beta:0F__inference_dense_4_layer_call_and_return_conditional_losses_129286698
UbS
	BiasAdd:0F__inference_dense_4_layer_call_and_return_conditional_losses_129286698
WbU
conv2d/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
ZbX
conv2d/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
YbW
conv2d_1/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
\bZ
conv2d_1/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
YbW
conv2d_2/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
\bZ
conv2d_2/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
YbW
conv2d_3/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
\bZ
conv2d_3/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
VbT
dense/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
YbW
dense/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
XbV
dense_1/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
[bY
dense_1/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
YbW
conv2d_4/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
\bZ
conv2d_4/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
XbV
dense_2/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
[bY
dense_2/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
XbV
dense_3/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
[bY
dense_3/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
XbV
dense_4/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287501
[bY
dense_4/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287501
WbU
conv2d/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
ZbX
conv2d/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
YbW
conv2d_1/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
\bZ
conv2d_1/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
YbW
conv2d_2/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
\bZ
conv2d_2/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
YbW
conv2d_3/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
\bZ
conv2d_3/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
VbT
dense/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
YbW
dense/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
XbV
dense_1/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
[bY
dense_1/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
YbW
conv2d_4/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
\bZ
conv2d_4/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
XbV
dense_2/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
[bY
dense_2/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
XbV
dense_3/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
[bY
dense_3/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
XbV
dense_4/beta:0D__inference_model_layer_call_and_return_conditional_losses_129287660
[bY
dense_4/BiasAdd:0D__inference_model_layer_call_and_return_conditional_losses_129287660
QbO
beta:0E__inference_conv2d_layer_call_and_return_conditional_losses_129287739
TbR
	BiasAdd:0E__inference_conv2d_layer_call_and_return_conditional_losses_129287739
SbQ
beta:0G__inference_conv2d_1_layer_call_and_return_conditional_losses_129287766
VbT
	BiasAdd:0G__inference_conv2d_1_layer_call_and_return_conditional_losses_129287766
SbQ
beta:0G__inference_conv2d_2_layer_call_and_return_conditional_losses_129287803
VbT
	BiasAdd:0G__inference_conv2d_2_layer_call_and_return_conditional_losses_129287803
SbQ
beta:0G__inference_conv2d_3_layer_call_and_return_conditional_losses_129287830
VbT
	BiasAdd:0G__inference_conv2d_3_layer_call_and_return_conditional_losses_129287830
PbN
beta:0D__inference_dense_layer_call_and_return_conditional_losses_129287867
SbQ
	BiasAdd:0D__inference_dense_layer_call_and_return_conditional_losses_129287867
SbQ
beta:0G__inference_conv2d_4_layer_call_and_return_conditional_losses_129287894
VbT
	BiasAdd:0G__inference_conv2d_4_layer_call_and_return_conditional_losses_129287894
RbP
beta:0F__inference_dense_1_layer_call_and_return_conditional_losses_129287921
UbS
	BiasAdd:0F__inference_dense_1_layer_call_and_return_conditional_losses_129287921
RbP
beta:0F__inference_dense_2_layer_call_and_return_conditional_losses_129287983
UbS
	BiasAdd:0F__inference_dense_2_layer_call_and_return_conditional_losses_129287983
RbP
beta:0F__inference_dense_3_layer_call_and_return_conditional_losses_129288010
UbS
	BiasAdd:0F__inference_dense_3_layer_call_and_return_conditional_losses_129288010
RbP
beta:0F__inference_dense_4_layer_call_and_return_conditional_losses_129288037
UbS
	BiasAdd:0F__inference_dense_4_layer_call_and_return_conditional_losses_129288037�
$__inference__wrapped_model_129286404�&'67?@PQbcYZ��������`�]
V�S
Q�N
)�&
input_1���������		
!�
input_2���������
� "1�.
,
dense_5!�
dense_5����������
J__inference_concatenate_layer_call_and_return_conditional_losses_129287956�[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1��������� 
� "&�#
�
0����������
� �
/__inference_concatenate_layer_call_fn_129287949x[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1��������� 
� "������������
G__inference_conv2d_1_layer_call_and_return_conditional_losses_129287766n&'8�5
.�+
)�&
inputs���������		�
� ".�+
$�!
0���������		�
� �
,__inference_conv2d_1_layer_call_fn_129287748a&'8�5
.�+
)�&
inputs���������		�
� "!����������		��
G__inference_conv2d_2_layer_call_and_return_conditional_losses_129287803n678�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_conv2d_2_layer_call_fn_129287785a678�5
.�+
)�&
inputs����������
� "!������������
G__inference_conv2d_3_layer_call_and_return_conditional_losses_129287830m?@8�5
.�+
)�&
inputs����������
� "-�*
#� 
0���������@
� �
,__inference_conv2d_3_layer_call_fn_129287812`?@8�5
.�+
)�&
inputs����������
� " ����������@�
G__inference_conv2d_4_layer_call_and_return_conditional_losses_129287894lYZ7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_conv2d_4_layer_call_fn_129287876_YZ7�4
-�*
(�%
inputs���������@
� " ����������@�
E__inference_conv2d_layer_call_and_return_conditional_losses_129287739m7�4
-�*
(�%
inputs���������		
� ".�+
$�!
0���������		�
� �
*__inference_conv2d_layer_call_fn_129287721`7�4
-�*
(�%
inputs���������		
� "!����������		��
F__inference_dense_1_layer_call_and_return_conditional_losses_129287921\bc/�,
%�"
 �
inputs���������@
� "%�"
�
0��������� 
� ~
+__inference_dense_1_layer_call_fn_129287903Obc/�,
%�"
 �
inputs���������@
� "���������� �
F__inference_dense_2_layer_call_and_return_conditional_losses_129287983`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_2_layer_call_fn_129287965S��0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_3_layer_call_and_return_conditional_losses_129288010`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_3_layer_call_fn_129287992S��0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_4_layer_call_and_return_conditional_losses_129288037_��0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
+__inference_dense_4_layer_call_fn_129288019R��0�-
&�#
!�
inputs����������
� "����������@�
F__inference_dense_5_layer_call_and_return_conditional_losses_129288056^��/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� �
+__inference_dense_5_layer_call_fn_129288046Q��/�,
%�"
 �
inputs���������@
� "�����������
D__inference_dense_layer_call_and_return_conditional_losses_129287867\PQ/�,
%�"
 �
inputs���������
� "%�"
�
0���������@
� |
)__inference_dense_layer_call_fn_129287849OPQ/�,
%�"
 �
inputs���������
� "����������@�
H__inference_flatten_1_layer_call_and_return_conditional_losses_129287943X/�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� |
-__inference_flatten_1_layer_call_fn_129287937K/�,
%�"
 �
inputs��������� 
� "���������� �
F__inference_flatten_layer_call_and_return_conditional_losses_129287932a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������
� �
+__inference_flatten_layer_call_fn_129287926T7�4
-�*
(�%
inputs���������@
� "������������
&__inference_internal_grad_fn_129288123���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288141���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288159���w�t
m�j

 
1�.
result_grads_0����������
1�.
result_grads_1����������
� "-�*

 
$�!
1�����������
&__inference_internal_grad_fn_129288177���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288195���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288213���e�b
[�X

 
(�%
result_grads_0��������� 
(�%
result_grads_1��������� 
� "$�!

 
�
1��������� �
&__inference_internal_grad_fn_129288231���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288249���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288267���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288285���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288303���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288321���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288339���w�t
m�j

 
1�.
result_grads_0����������
1�.
result_grads_1����������
� "-�*

 
$�!
1�����������
&__inference_internal_grad_fn_129288357���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288375���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288393���e�b
[�X

 
(�%
result_grads_0��������� 
(�%
result_grads_1��������� 
� "$�!

 
�
1��������� �
&__inference_internal_grad_fn_129288411���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288429���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288447���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288465���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288483���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288501���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288519���w�t
m�j

 
1�.
result_grads_0����������
1�.
result_grads_1����������
� "-�*

 
$�!
1�����������
&__inference_internal_grad_fn_129288537���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288555���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288573���e�b
[�X

 
(�%
result_grads_0��������� 
(�%
result_grads_1��������� 
� "$�!

 
�
1��������� �
&__inference_internal_grad_fn_129288591���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288609���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288627���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288645���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288663���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288681���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288699���w�t
m�j

 
1�.
result_grads_0����������
1�.
result_grads_1����������
� "-�*

 
$�!
1�����������
&__inference_internal_grad_fn_129288717���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288735���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288753���e�b
[�X

 
(�%
result_grads_0��������� 
(�%
result_grads_1��������� 
� "$�!

 
�
1��������� �
&__inference_internal_grad_fn_129288771���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288789���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288807���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288825���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288843���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288861���w�t
m�j

 
1�.
result_grads_0���������		�
1�.
result_grads_1���������		�
� "-�*

 
$�!
1���������		��
&__inference_internal_grad_fn_129288879���w�t
m�j

 
1�.
result_grads_0����������
1�.
result_grads_1����������
� "-�*

 
$�!
1�����������
&__inference_internal_grad_fn_129288897���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288915���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
&__inference_internal_grad_fn_129288933���u�r
k�h

 
0�-
result_grads_0���������@
0�-
result_grads_1���������@
� ",�)

 
#� 
1���������@�
&__inference_internal_grad_fn_129288951���e�b
[�X

 
(�%
result_grads_0��������� 
(�%
result_grads_1��������� 
� "$�!

 
�
1��������� �
&__inference_internal_grad_fn_129288969���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129288987���g�d
]�Z

 
)�&
result_grads_0����������
)�&
result_grads_1����������
� "%�"

 
�
1�����������
&__inference_internal_grad_fn_129289005���e�b
[�X

 
(�%
result_grads_0���������@
(�%
result_grads_1���������@
� "$�!

 
�
1���������@�
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_129287840�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
3__inference_max_pooling2d_1_layer_call_fn_129287835�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_129287776�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_max_pooling2d_layer_call_fn_129287771�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_model_layer_call_and_return_conditional_losses_129287177�&'67?@PQbcYZ��������h�e
^�[
Q�N
)�&
input_1���������		
!�
input_2���������
p 

 
� "%�"
�
0���������
� �
D__inference_model_layer_call_and_return_conditional_losses_129287242�&'67?@PQbcYZ��������h�e
^�[
Q�N
)�&
input_1���������		
!�
input_2���������
p

 
� "%�"
�
0���������
� �
D__inference_model_layer_call_and_return_conditional_losses_129287501�&'67?@PQbcYZ��������j�g
`�]
S�P
*�'
inputs/0���������		
"�
inputs/1���������
p 

 
� "%�"
�
0���������
� �
D__inference_model_layer_call_and_return_conditional_losses_129287660�&'67?@PQbcYZ��������j�g
`�]
S�P
*�'
inputs/0���������		
"�
inputs/1���������
p

 
� "%�"
�
0���������
� �
)__inference_model_layer_call_fn_129286768�&'67?@PQbcYZ��������h�e
^�[
Q�N
)�&
input_1���������		
!�
input_2���������
p 

 
� "�����������
)__inference_model_layer_call_fn_129287112�&'67?@PQbcYZ��������h�e
^�[
Q�N
)�&
input_1���������		
!�
input_2���������
p

 
� "�����������
)__inference_model_layer_call_fn_129287292�&'67?@PQbcYZ��������j�g
`�]
S�P
*�'
inputs/0���������		
"�
inputs/1���������
p 

 
� "�����������
)__inference_model_layer_call_fn_129287342�&'67?@PQbcYZ��������j�g
`�]
S�P
*�'
inputs/0���������		
"�
inputs/1���������
p

 
� "�����������
'__inference_signature_wrapper_129287712�&'67?@PQbcYZ��������q�n
� 
g�d
4
input_1)�&
input_1���������		
,
input_2!�
input_2���������"1�.
,
dense_5!�
dense_5���������