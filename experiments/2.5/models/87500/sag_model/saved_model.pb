НЌ
Ђѓ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
С
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
executor_typestring Ј
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68­§

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
~
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*'
_output_shapes
:*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel

$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*(
_output_shapes
:*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_13/kernel
~
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*'
_output_shapes
:@*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:@*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:@*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:@*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@ *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
: *
dtype0
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 * 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
 *
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:*
dtype0
|
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_15/kernel
u
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel* 
_output_shapes
:
*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:*
dtype0
{
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_16/kernel
t
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes
:	@*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:@*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ъc
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*c
valueћbBјb Bёb
ѓ
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
Ы

kernel
bias
#_self_saveable_object_factories
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
Ы

&kernel
'bias
#(_self_saveable_object_factories
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
Г
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
Ы

6kernel
7bias
#8_self_saveable_object_factories
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
Ы

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
Г
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
Ы

Pkernel
Qbias
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*
Ы

Ykernel
Zbias
#[_self_saveable_object_factories
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*
Ы

bkernel
cbias
#d_self_saveable_object_factories
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses*
Г
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
Г
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses* 
Г
#y_self_saveable_object_factories
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
д
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
д
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
д
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
д
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
 regularization_losses
Ё	keras_api
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses*

Єserving_default* 
* 
В
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
14
15
16
17
18
19
20
21*
В
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
14
15
16
17
18
19
20
21*
* 
Е
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
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
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

&0
'1*

&0
'1*
* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
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

Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

60
71*

60
71*
* 

Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
@1*

?0
@1*
* 

Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
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

Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

P0
Q1*

P0
Q1*
* 

Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Y0
Z1*

Y0
Z1*
* 

Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

b0
c1*

b0
c1*
* 

вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
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

зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
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

мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
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

сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_16/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEdense_17/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_17/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
	variables
trainable_variables
 regularization_losses
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses*
* 
* 
* 
* 

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

serving_default_input_5Placeholder*/
_output_shapes
:џџџџџџџџџ		*
dtype0*$
shape:џџџџџџџџџ		
z
serving_default_input_6Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
ъ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_6conv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasconv2d_14/kernelconv2d_14/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *0
f+R)
'__inference_signature_wrapper_139120742
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
л
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpConst*#
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
GPU2*0,1J 8 *+
f&R$
"__inference__traced_save_139122076

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_12/kerneldense_12/biasconv2d_14/kernelconv2d_14/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*"
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
GPU2*0,1J 8 *.
f)R'
%__inference__traced_restore_139122152ой
яJ
г

F__inference_model_2_layer_call_and_return_conditional_losses_139119751

inputs
inputs_1.
conv2d_10_139119486:"
conv2d_10_139119488:	/
conv2d_11_139119510:"
conv2d_11_139119512:	/
conv2d_12_139119535:"
conv2d_12_139119537:	.
conv2d_13_139119559:@!
conv2d_13_139119561:@$
dense_12_139119583:@ 
dense_12_139119585:@$
dense_13_139119608:@  
dense_13_139119610: -
conv2d_14_139119632:@@!
conv2d_14_139119634:@&
dense_14_139119681:
 !
dense_14_139119683:	&
dense_15_139119705:
!
dense_15_139119707:	%
dense_16_139119729:	@ 
dense_16_139119731:@$
dense_17_139119745:@ 
dense_17_139119747:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_139119486conv2d_10_139119488*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485Џ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_139119510conv2d_11_139119512*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509ћ
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139119443­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_139119535conv2d_12_139119537*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534Ў
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_139119559conv2d_13_139119561*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_139119583dense_12_139119585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_139119582њ
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139119455Ё
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_139119608dense_13_139119610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_139119607Ќ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_139119632conv2d_14_139119634*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631ч
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_139119643х
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_139119651
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139119660
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_139119681dense_14_139119683*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_139119680Ђ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_139119705dense_15_139119707*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_139119704Ё
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_139119729dense_16_139119731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_139119728Ё
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_139119745dense_17_139119747*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_139119744x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЬ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
н

F__inference_model_2_layer_call_and_return_conditional_losses_139120531
inputs_0
inputs_1C
(conv2d_10_conv2d_readvariableop_resource:8
)conv2d_10_biasadd_readvariableop_resource:	D
(conv2d_11_conv2d_readvariableop_resource:8
)conv2d_11_biasadd_readvariableop_resource:	D
(conv2d_12_conv2d_readvariableop_resource:8
)conv2d_12_biasadd_readvariableop_resource:	C
(conv2d_13_conv2d_readvariableop_resource:@7
)conv2d_13_biasadd_readvariableop_resource:@9
'dense_12_matmul_readvariableop_resource:@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@ 6
(dense_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource:@@7
)conv2d_14_biasadd_readvariableop_resource:@;
'dense_14_matmul_readvariableop_resource:
 7
(dense_14_biasadd_readvariableop_resource:	;
'dense_15_matmul_readvariableop_resource:
7
(dense_15_biasadd_readvariableop_resource:	:
'dense_16_matmul_readvariableop_resource:	@6
(dense_16_biasadd_readvariableop_resource:@9
'dense_17_matmul_readvariableop_resource:@6
(dense_17_biasadd_readvariableop_resource:
identityЂ conv2d_10/BiasAdd/ReadVariableOpЂconv2d_10/Conv2D/ReadVariableOpЂ conv2d_11/BiasAdd/ReadVariableOpЂconv2d_11/Conv2D/ReadVariableOpЂ conv2d_12/BiasAdd/ReadVariableOpЂconv2d_12/Conv2D/ReadVariableOpЂ conv2d_13/BiasAdd/ReadVariableOpЂconv2d_13/Conv2D/ReadVariableOpЂ conv2d_14/BiasAdd/ReadVariableOpЂconv2d_14/Conv2D/ReadVariableOpЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOpЂdense_14/BiasAdd/ReadVariableOpЂdense_14/MatMul/ReadVariableOpЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpЂdense_16/BiasAdd/ReadVariableOpЂdense_16/MatMul/ReadVariableOpЂdense_17/BiasAdd/ReadVariableOpЂdense_17/MatMul/ReadVariableOp
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0А
conv2d_10/Conv2DConv2Dinputs_0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		S
conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_10/mulMulconv2d_10/beta:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		j
conv2d_10/SigmoidSigmoidconv2d_10/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
conv2d_10/mul_1Mulconv2d_10/BiasAdd:output:0conv2d_10/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		n
conv2d_10/IdentityIdentityconv2d_10/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		н
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120383*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ф
conv2d_11/Conv2DConv2Dconv2d_10/IdentityN:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		S
conv2d_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_11/mulMulconv2d_11/beta:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		j
conv2d_11/SigmoidSigmoidconv2d_11/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
conv2d_11/mul_1Mulconv2d_11/BiasAdd:output:0conv2d_11/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		n
conv2d_11/IdentityIdentityconv2d_11/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		н
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120397*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		Ў
max_pooling2d_4/MaxPoolMaxPoolconv2d_11/IdentityN:output:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ш
conv2d_12/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџS
conv2d_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_12/mulMulconv2d_12/beta:output:0conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
conv2d_12/SigmoidSigmoidconv2d_12/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_12/mul_1Mulconv2d_12/BiasAdd:output:0conv2d_12/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџn
conv2d_12/IdentityIdentityconv2d_12/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџн
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120412*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0У
conv2d_13/Conv2DConv2Dconv2d_12/IdentityN:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@S
conv2d_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_13/mulMulconv2d_13/beta:output:0conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
conv2d_13/SigmoidSigmoidconv2d_13/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
conv2d_13/mul_1Mulconv2d_13/BiasAdd:output:0conv2d_13/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@m
conv2d_13/IdentityIdentityconv2d_13/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@л
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120426*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0}
dense_12/MatMulMatMulinputs_1&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_12/mulMuldense_12/beta:output:0dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
dense_12/SigmoidSigmoiddense_12/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
dense_12/mul_1Muldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_12/IdentityIdentitydense_12/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ш
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120440*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@­
max_pooling2d_5/MaxPoolMaxPoolconv2d_13/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_13/MatMulMatMuldense_12/IdentityN:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ R
dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_13/mulMuldense_13/beta:output:0dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
dense_13/SigmoidSigmoiddense_13/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
dense_13/mul_1Muldense_13/BiasAdd:output:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
dense_13/IdentityIdentitydense_13/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Ш
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120455*:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ 
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ч
conv2d_14/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@S
conv2d_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_14/mulMulconv2d_14/beta:output:0conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
conv2d_14/SigmoidSigmoidconv2d_14/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
conv2d_14/mul_1Mulconv2d_14/BiasAdd:output:0conv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@m
conv2d_14/IdentityIdentityconv2d_14/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@л
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120469*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_4/ReshapeReshapeconv2d_14/IdentityN:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
flatten_5/ReshapeReshapedense_13/IdentityN:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :И
concatenate_2/concatConcatV2flatten_4/Reshape:output:0flatten_5/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_14/MatMulMatMulconcatenate_2/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџR
dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_14/mulMuldense_14/beta:output:0dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ`
dense_14/SigmoidSigmoiddense_14/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџy
dense_14/mul_1Muldense_14/BiasAdd:output:0dense_14/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_14/IdentityIdentitydense_14/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120489*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_15/MatMulMatMuldense_14/IdentityN:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџR
dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_15/mulMuldense_15/beta:output:0dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ`
dense_15/SigmoidSigmoiddense_15/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџy
dense_15/mul_1Muldense_15/BiasAdd:output:0dense_15/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_15/IdentityIdentitydense_15/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120503*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_16/MatMulMatMuldense_15/IdentityN:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_16/mulMuldense_16/beta:output:0dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
dense_16/SigmoidSigmoiddense_16/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
dense_16/mul_1Muldense_16/BiasAdd:output:0dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_16/IdentityIdentitydense_16/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ш
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120517*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_17/MatMulMatMuldense_16/IdentityN:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџБ
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
/
_output_shapes
:џџџџџџџџџ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ф

&__inference_internal_grad_fn_139121711
result_grads_0
result_grads_1
mul_conv2d_11_beta
mul_conv2d_11_biasadd
identity
mulMulmul_conv2d_11_betamul_conv2d_11_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		r
mul_1Mulmul_conv2d_11_betamul_conv2d_11_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		
х

&__inference_internal_grad_fn_139121639
result_grads_0
result_grads_1
mul_dense_14_beta
mul_dense_14_biasadd
identityw
mulMulmul_dense_14_betamul_dense_14_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџh
mul_1Mulmul_dense_14_betamul_dense_14_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
Ў
}
&__inference_internal_grad_fn_139121459
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
Ъ
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_139119643

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ў
}
&__inference_internal_grad_fn_139121999
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
ф

&__inference_internal_grad_fn_139121531
result_grads_0
result_grads_1
mul_conv2d_11_beta
mul_conv2d_11_biasadd
identity
mulMulmul_conv2d_11_betamul_conv2d_11_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		r
mul_1Mulmul_conv2d_11_betamul_conv2d_11_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		
ёJ
г

F__inference_model_2_layer_call_and_return_conditional_losses_139120272
input_5
input_6.
conv2d_10_139120211:"
conv2d_10_139120213:	/
conv2d_11_139120216:"
conv2d_11_139120218:	/
conv2d_12_139120222:"
conv2d_12_139120224:	.
conv2d_13_139120227:@!
conv2d_13_139120229:@$
dense_12_139120232:@ 
dense_12_139120234:@$
dense_13_139120238:@  
dense_13_139120240: -
conv2d_14_139120243:@@!
conv2d_14_139120245:@&
dense_14_139120251:
 !
dense_14_139120253:	&
dense_15_139120256:
!
dense_15_139120258:	%
dense_16_139120261:	@ 
dense_16_139120263:@$
dense_17_139120266:@ 
dense_17_139120268:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_139120211conv2d_10_139120213*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485Џ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_139120216conv2d_11_139120218*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509ћ
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139119443­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_139120222conv2d_12_139120224*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534Ў
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_139120227conv2d_13_139120229*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558џ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_139120232dense_12_139120234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_139119582њ
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139119455Ё
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_139120238dense_13_139120240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_139119607Ќ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_139120243conv2d_14_139120245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631ч
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_139119643х
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_139119651
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139119660
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_139120251dense_14_139120253*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_139119680Ђ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_139120256dense_15_139120258*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_139119704Ё
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_139120261dense_16_139120263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_139119728Ё
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_139120266dense_17_139120268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_139119744x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЬ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
Э

,__inference_dense_17_layer_call_fn_139121076

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_139119744o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ъ

'__inference_signature_wrapper_139120742
input_5
input_6"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
 

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	@

unknown_18:@

unknown_19:@

unknown_20:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *-
f(R&
$__inference__wrapped_model_139119434o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
ф

&__inference_internal_grad_fn_139121729
result_grads_0
result_grads_1
mul_conv2d_12_beta
mul_conv2d_12_biasadd
identity
mulMulmul_conv2d_12_betamul_conv2d_12_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџr
mul_1Mulmul_conv2d_12_betamul_conv2d_12_biasadd*
T0*0
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:` \
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ
Ъ	
ј
G__inference_dense_17_layer_call_and_return_conditional_losses_139119744

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Њ
I
-__inference_flatten_5_layer_call_fn_139120967

inputs
identityИ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_139119651`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Э

H__inference_conv2d_10_layer_call_and_return_conditional_losses_139120769

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		П
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120761*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs
Э

H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		П
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119477*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs
Ъ	
ј
G__inference_dense_17_layer_call_and_return_conditional_losses_139121086

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ї
Ђ
-__inference_conv2d_14_layer_call_fn_139120906

inputs!
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
б

H__inference_conv2d_11_layer_call_and_return_conditional_losses_139120796

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		П
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120788*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs
Г
ћ
G__inference_dense_16_layer_call_and_return_conditional_losses_139119728

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119720*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж

&__inference_internal_grad_fn_139121675
result_grads_0
result_grads_1
mul_dense_16_beta
mul_dense_16_biasadd
identityv
mulMulmul_dense_16_betamul_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
mul_1Mulmul_dense_16_betamul_dense_16_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@

}
&__inference_internal_grad_fn_139121495
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@
Р
§
G__inference_dense_14_layer_call_and_return_conditional_losses_139121013

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЏ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139121005*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

Ё
&__inference_internal_grad_fn_139121207
result_grads_0
result_grads_1
mul_model_2_conv2d_13_beta!
mul_model_2_conv2d_13_biasadd
identity
mulMulmul_model_2_conv2d_13_betamul_model_2_conv2d_13_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
mul_1Mulmul_model_2_conv2d_13_betamul_model_2_conv2d_13_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
І
}
&__inference_internal_grad_fn_139121369
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:` \
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ
Э

,__inference_dense_13_layer_call_fn_139120933

inputs
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_139119607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
№

+__inference_model_2_layer_call_fn_139119798
input_5
input_6"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
 

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	@

unknown_18:@

unknown_19:@

unknown_20:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_139119751o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
д

,__inference_dense_15_layer_call_fn_139121022

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_139119704p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

}
&__inference_internal_grad_fn_139121927
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
ў
Ѕ
-__inference_conv2d_11_layer_call_fn_139120778

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ		: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs
Џ
њ
G__inference_dense_12_layer_call_and_return_conditional_losses_139120897

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120889*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


&__inference_internal_grad_fn_139121297
result_grads_0
result_grads_1
mul_model_2_dense_15_beta 
mul_model_2_dense_15_biasadd
identity
mulMulmul_model_2_dense_15_betamul_model_2_dense_15_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџx
mul_1Mulmul_model_2_dense_15_betamul_model_2_dense_15_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
І
}
&__inference_internal_grad_fn_139121909
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:` \
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ
а

,__inference_dense_16_layer_call_fn_139121049

inputs
unknown:	@
	unknown_0:@
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_139119728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ћ
Є
-__inference_conv2d_10_layer_call_fn_139120751

inputs"
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ		: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs

}
&__inference_internal_grad_fn_139121963
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@

}
&__inference_internal_grad_fn_139122035
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@
ф

&__inference_internal_grad_fn_139121513
result_grads_0
result_grads_1
mul_conv2d_10_beta
mul_conv2d_10_biasadd
identity
mulMulmul_conv2d_10_betamul_conv2d_10_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		r
mul_1Mulmul_conv2d_10_betamul_conv2d_10_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		
ф

&__inference_internal_grad_fn_139121549
result_grads_0
result_grads_1
mul_conv2d_12_beta
mul_conv2d_12_biasadd
identity
mulMulmul_conv2d_12_betamul_conv2d_12_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџr
mul_1Mulmul_conv2d_12_betamul_conv2d_12_biasadd*
T0*0
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:` \
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ

}
&__inference_internal_grad_fn_139121405
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@
е

&__inference_internal_grad_fn_139121747
result_grads_0
result_grads_1
mul_conv2d_13_beta
mul_conv2d_13_biasadd
identity
mulMulmul_conv2d_13_betamul_conv2d_13_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@q
mul_1Mulmul_conv2d_13_betamul_conv2d_13_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
Џ
њ
G__inference_dense_13_layer_call_and_return_conditional_losses_139119607

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119599*:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
д

,__inference_dense_14_layer_call_fn_139120995

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_139119680p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Р
§
G__inference_dense_14_layer_call_and_return_conditional_losses_139119680

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЏ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119672*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
У
O
3__inference_max_pooling2d_5_layer_call_fn_139120865

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139119455
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139119455

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Р

H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
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
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Н
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119623*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
№

+__inference_model_2_layer_call_fn_139120142
input_5
input_6"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
 

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	@

unknown_18:@

unknown_19:@

unknown_20:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_139120045o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
С
v
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139119660

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
:џџџџџџџџџ X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ:џџџџџџџџџ :P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


&__inference_internal_grad_fn_139121315
result_grads_0
result_grads_1
mul_model_2_dense_16_beta 
mul_model_2_dense_16_biasadd
identity
mulMulmul_model_2_dense_16_betamul_model_2_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
mul_1Mulmul_model_2_dense_16_betamul_model_2_dense_16_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@
ў
Ѕ
-__inference_conv2d_12_layer_call_fn_139120815

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
њ
G__inference_dense_12_layer_call_and_return_conditional_losses_139119582

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119574*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ё
&__inference_internal_grad_fn_139121171
result_grads_0
result_grads_1
mul_model_2_conv2d_11_beta!
mul_model_2_conv2d_11_biasadd
identity
mulMulmul_model_2_conv2d_11_betamul_model_2_conv2d_11_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
mul_1Mulmul_model_2_conv2d_11_betamul_model_2_conv2d_11_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		

Ё
&__inference_internal_grad_fn_139121189
result_grads_0
result_grads_1
mul_model_2_conv2d_12_beta!
mul_model_2_conv2d_12_biasadd
identity
mulMulmul_model_2_conv2d_12_betamul_model_2_conv2d_12_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ
mul_1Mulmul_model_2_conv2d_12_betamul_model_2_conv2d_12_biasadd*
T0*0
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:` \
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ
е

&__inference_internal_grad_fn_139121621
result_grads_0
result_grads_1
mul_conv2d_14_beta
mul_conv2d_14_biasadd
identity
mulMulmul_conv2d_14_betamul_conv2d_14_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@q
mul_1Mulmul_conv2d_14_betamul_conv2d_14_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
ж

&__inference_internal_grad_fn_139121585
result_grads_0
result_grads_1
mul_dense_12_beta
mul_dense_12_biasadd
identityv
mulMulmul_dense_12_betamul_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
mul_1Mulmul_dense_12_betamul_dense_12_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@
Р
§
G__inference_dense_15_layer_call_and_return_conditional_losses_139121040

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЏ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139121032*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


&__inference_internal_grad_fn_139121225
result_grads_0
result_grads_1
mul_model_2_dense_12_beta 
mul_model_2_dense_12_biasadd
identity
mulMulmul_model_2_dense_12_betamul_model_2_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
mul_1Mulmul_model_2_dense_12_betamul_model_2_dense_12_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@

j
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139120806

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


&__inference_internal_grad_fn_139121279
result_grads_0
result_grads_1
mul_model_2_dense_14_beta 
mul_model_2_dense_14_biasadd
identity
mulMulmul_model_2_dense_14_betamul_model_2_dense_14_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџx
mul_1Mulmul_model_2_dense_14_betamul_model_2_dense_14_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
ёJ
г

F__inference_model_2_layer_call_and_return_conditional_losses_139120207
input_5
input_6.
conv2d_10_139120146:"
conv2d_10_139120148:	/
conv2d_11_139120151:"
conv2d_11_139120153:	/
conv2d_12_139120157:"
conv2d_12_139120159:	.
conv2d_13_139120162:@!
conv2d_13_139120164:@$
dense_12_139120167:@ 
dense_12_139120169:@$
dense_13_139120173:@  
dense_13_139120175: -
conv2d_14_139120178:@@!
conv2d_14_139120180:@&
dense_14_139120186:
 !
dense_14_139120188:	&
dense_15_139120191:
!
dense_15_139120193:	%
dense_16_139120196:	@ 
dense_16_139120198:@$
dense_17_139120201:@ 
dense_17_139120203:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_139120146conv2d_10_139120148*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485Џ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_139120151conv2d_11_139120153*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509ћ
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139119443­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_139120157conv2d_12_139120159*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534Ў
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_139120162conv2d_13_139120164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558џ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_139120167dense_12_139120169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_139119582њ
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139119455Ё
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_139120173dense_13_139120175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_139119607Ќ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_139120178conv2d_14_139120180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631ч
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_139119643х
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_139119651
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139119660
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_139120186dense_14_139120188*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_139119680Ђ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_139120191dense_15_139120193*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_139119704Ё
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_139120196dense_16_139120198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_139119728Ё
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_139120201dense_17_139120203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_139119744x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЬ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
Р

H__inference_conv2d_14_layer_call_and_return_conditional_losses_139120924

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
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
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Н
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120916*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ЂX
в
%__inference__traced_restore_139122152
file_prefix<
!assignvariableop_conv2d_10_kernel:0
!assignvariableop_1_conv2d_10_bias:	?
#assignvariableop_2_conv2d_11_kernel:0
!assignvariableop_3_conv2d_11_bias:	?
#assignvariableop_4_conv2d_12_kernel:0
!assignvariableop_5_conv2d_12_bias:	>
#assignvariableop_6_conv2d_13_kernel:@/
!assignvariableop_7_conv2d_13_bias:@4
"assignvariableop_8_dense_12_kernel:@.
 assignvariableop_9_dense_12_bias:@>
$assignvariableop_10_conv2d_14_kernel:@@0
"assignvariableop_11_conv2d_14_bias:@5
#assignvariableop_12_dense_13_kernel:@ /
!assignvariableop_13_dense_13_bias: 7
#assignvariableop_14_dense_14_kernel:
 0
!assignvariableop_15_dense_14_bias:	7
#assignvariableop_16_dense_15_kernel:
0
!assignvariableop_17_dense_15_bias:	6
#assignvariableop_18_dense_16_kernel:	@/
!assignvariableop_19_dense_16_bias:@5
#assignvariableop_20_dense_17_kernel:@/
!assignvariableop_21_dense_17_bias:
identity_23ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Щ

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*я	
valueх	Bт	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_12_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_12_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_14_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_14_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_13_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_13_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_14_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_14_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_15_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_15_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_16_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_16_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_17_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_17_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Г
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
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
І
}
&__inference_internal_grad_fn_139121351
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		
Ў
}
&__inference_internal_grad_fn_139122017
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
ж

&__inference_internal_grad_fn_139121855
result_grads_0
result_grads_1
mul_dense_16_beta
mul_dense_16_biasadd
identityv
mulMulmul_dense_16_betamul_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
mul_1Mulmul_dense_16_betamul_dense_16_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@
И
d
H__inference_flatten_5_layer_call_and_return_conditional_losses_139119651

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Џ
њ
G__inference_dense_13_layer_call_and_return_conditional_losses_139120951

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120943*:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139119443

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б

H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџП
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119526*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџl

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж

&__inference_internal_grad_fn_139121783
result_grads_0
result_grads_1
mul_dense_13_beta
mul_dense_13_biasadd
identityv
mulMulmul_dense_13_betamul_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ g
mul_1Mulmul_dense_13_betamul_dense_13_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :W S
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ 
У
O
3__inference_max_pooling2d_4_layer_call_fn_139120801

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139119443
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ф

H__inference_conv2d_13_layer_call_and_return_conditional_losses_139120860

inputs9
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
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
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Н
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120852*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
d
H__inference_flatten_5_layer_call_and_return_conditional_losses_139120973

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

}
&__inference_internal_grad_fn_139121441
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
н

F__inference_model_2_layer_call_and_return_conditional_losses_139120690
inputs_0
inputs_1C
(conv2d_10_conv2d_readvariableop_resource:8
)conv2d_10_biasadd_readvariableop_resource:	D
(conv2d_11_conv2d_readvariableop_resource:8
)conv2d_11_biasadd_readvariableop_resource:	D
(conv2d_12_conv2d_readvariableop_resource:8
)conv2d_12_biasadd_readvariableop_resource:	C
(conv2d_13_conv2d_readvariableop_resource:@7
)conv2d_13_biasadd_readvariableop_resource:@9
'dense_12_matmul_readvariableop_resource:@6
(dense_12_biasadd_readvariableop_resource:@9
'dense_13_matmul_readvariableop_resource:@ 6
(dense_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource:@@7
)conv2d_14_biasadd_readvariableop_resource:@;
'dense_14_matmul_readvariableop_resource:
 7
(dense_14_biasadd_readvariableop_resource:	;
'dense_15_matmul_readvariableop_resource:
7
(dense_15_biasadd_readvariableop_resource:	:
'dense_16_matmul_readvariableop_resource:	@6
(dense_16_biasadd_readvariableop_resource:@9
'dense_17_matmul_readvariableop_resource:@6
(dense_17_biasadd_readvariableop_resource:
identityЂ conv2d_10/BiasAdd/ReadVariableOpЂconv2d_10/Conv2D/ReadVariableOpЂ conv2d_11/BiasAdd/ReadVariableOpЂconv2d_11/Conv2D/ReadVariableOpЂ conv2d_12/BiasAdd/ReadVariableOpЂconv2d_12/Conv2D/ReadVariableOpЂ conv2d_13/BiasAdd/ReadVariableOpЂconv2d_13/Conv2D/ReadVariableOpЂ conv2d_14/BiasAdd/ReadVariableOpЂconv2d_14/Conv2D/ReadVariableOpЂdense_12/BiasAdd/ReadVariableOpЂdense_12/MatMul/ReadVariableOpЂdense_13/BiasAdd/ReadVariableOpЂdense_13/MatMul/ReadVariableOpЂdense_14/BiasAdd/ReadVariableOpЂdense_14/MatMul/ReadVariableOpЂdense_15/BiasAdd/ReadVariableOpЂdense_15/MatMul/ReadVariableOpЂdense_16/BiasAdd/ReadVariableOpЂdense_16/MatMul/ReadVariableOpЂdense_17/BiasAdd/ReadVariableOpЂdense_17/MatMul/ReadVariableOp
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0А
conv2d_10/Conv2DConv2Dinputs_0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		S
conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_10/mulMulconv2d_10/beta:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		j
conv2d_10/SigmoidSigmoidconv2d_10/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
conv2d_10/mul_1Mulconv2d_10/BiasAdd:output:0conv2d_10/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		n
conv2d_10/IdentityIdentityconv2d_10/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		н
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120542*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ф
conv2d_11/Conv2DConv2Dconv2d_10/IdentityN:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		S
conv2d_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_11/mulMulconv2d_11/beta:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		j
conv2d_11/SigmoidSigmoidconv2d_11/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
conv2d_11/mul_1Mulconv2d_11/BiasAdd:output:0conv2d_11/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		n
conv2d_11/IdentityIdentityconv2d_11/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		н
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120556*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		Ў
max_pooling2d_4/MaxPoolMaxPoolconv2d_11/IdentityN:output:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ш
conv2d_12/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџS
conv2d_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_12/mulMulconv2d_12/beta:output:0conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџj
conv2d_12/SigmoidSigmoidconv2d_12/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ
conv2d_12/mul_1Mulconv2d_12/BiasAdd:output:0conv2d_12/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџn
conv2d_12/IdentityIdentityconv2d_12/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџн
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120571*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0У
conv2d_13/Conv2DConv2Dconv2d_12/IdentityN:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@S
conv2d_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_13/mulMulconv2d_13/beta:output:0conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
conv2d_13/SigmoidSigmoidconv2d_13/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
conv2d_13/mul_1Mulconv2d_13/BiasAdd:output:0conv2d_13/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@m
conv2d_13/IdentityIdentityconv2d_13/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@л
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120585*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0}
dense_12/MatMulMatMulinputs_1&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_12/mulMuldense_12/beta:output:0dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
dense_12/SigmoidSigmoiddense_12/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
dense_12/mul_1Muldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_12/IdentityIdentitydense_12/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ш
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120599*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@­
max_pooling2d_5/MaxPoolMaxPoolconv2d_13/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_13/MatMulMatMuldense_12/IdentityN:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ R
dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_13/mulMuldense_13/beta:output:0dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ _
dense_13/SigmoidSigmoiddense_13/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ x
dense_13/mul_1Muldense_13/BiasAdd:output:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
dense_13/IdentityIdentitydense_13/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Ш
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120614*:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ 
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ч
conv2d_14/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@S
conv2d_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_14/mulMulconv2d_14/beta:output:0conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
conv2d_14/SigmoidSigmoidconv2d_14/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
conv2d_14/mul_1Mulconv2d_14/BiasAdd:output:0conv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@m
conv2d_14/IdentityIdentityconv2d_14/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@л
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120628*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_4/ReshapeReshapeconv2d_14/IdentityN:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
flatten_5/ReshapeReshapedense_13/IdentityN:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :И
concatenate_2/concatConcatV2flatten_4/Reshape:output:0flatten_5/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_14/MatMulMatMulconcatenate_2/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџR
dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_14/mulMuldense_14/beta:output:0dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ`
dense_14/SigmoidSigmoiddense_14/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџy
dense_14/mul_1Muldense_14/BiasAdd:output:0dense_14/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_14/IdentityIdentitydense_14/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120648*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_15/MatMulMatMuldense_14/IdentityN:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџR
dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_15/mulMuldense_15/beta:output:0dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ`
dense_15/SigmoidSigmoiddense_15/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџy
dense_15/mul_1Muldense_15/BiasAdd:output:0dense_15/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџd
dense_15/IdentityIdentitydense_15/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120662*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_16/MatMulMatMuldense_15/IdentityN:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_16/mulMuldense_16/beta:output:0dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
dense_16/SigmoidSigmoiddense_16/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
dense_16/mul_1Muldense_16/BiasAdd:output:0dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
dense_16/IdentityIdentitydense_16/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ш
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120676*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_17/MatMulMatMuldense_16/IdentityN:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџБ
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
/
_output_shapes
:џџџџџџџџџ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
І
}
&__inference_internal_grad_fn_139121333
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		

Ё
&__inference_internal_grad_fn_139121153
result_grads_0
result_grads_1
mul_model_2_conv2d_10_beta!
mul_model_2_conv2d_10_biasadd
identity
mulMulmul_model_2_conv2d_10_betamul_model_2_conv2d_10_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
mul_1Mulmul_model_2_conv2d_10_betamul_model_2_conv2d_10_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		
І
}
&__inference_internal_grad_fn_139121891
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		
њ
Ѓ
-__inference_conv2d_13_layer_call_fn_139120842

inputs"
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ё
&__inference_internal_grad_fn_139121261
result_grads_0
result_grads_1
mul_model_2_conv2d_14_beta!
mul_model_2_conv2d_14_biasadd
identity
mulMulmul_model_2_conv2d_14_betamul_model_2_conv2d_14_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
mul_1Mulmul_model_2_conv2d_14_betamul_model_2_conv2d_14_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@

}
&__inference_internal_grad_fn_139121981
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :W S
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ 
б

H__inference_conv2d_12_layer_call_and_return_conditional_losses_139120833

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџП
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139120825*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџl

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е

&__inference_internal_grad_fn_139121801
result_grads_0
result_grads_1
mul_conv2d_14_beta
mul_conv2d_14_biasadd
identity
mulMulmul_conv2d_14_betamul_conv2d_14_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@q
mul_1Mulmul_conv2d_14_betamul_conv2d_14_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
І
}
&__inference_internal_grad_fn_139121873
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		

}
&__inference_internal_grad_fn_139121387
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
і

+__inference_model_2_layer_call_fn_139120372
inputs_0
inputs_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
 

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	@

unknown_18:@

unknown_19:@

unknown_20:
identityЂStatefulPartitionedCallќ
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
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_139120045o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
б

H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		П
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119501*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs
і

+__inference_model_2_layer_call_fn_139120322
inputs_0
inputs_1"
unknown:
	unknown_0:	%
	unknown_1:
	unknown_2:	%
	unknown_3:
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: $

unknown_11:@@

unknown_12:@

unknown_13:
 

unknown_14:	

unknown_15:


unknown_16:	

unknown_17:	@

unknown_18:@

unknown_19:@

unknown_20:
identityЂStatefulPartitionedCallќ
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
:џџџџџџџџџ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_139119751o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1

}
&__inference_internal_grad_fn_139121423
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :W S
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ 


&__inference_internal_grad_fn_139121243
result_grads_0
result_grads_1
mul_model_2_dense_13_beta 
mul_model_2_dense_13_biasadd
identity
mulMulmul_model_2_dense_13_betamul_model_2_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ w
mul_1Mulmul_model_2_dense_13_betamul_model_2_dense_13_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :W S
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ 

j
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139120870

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
М
I
-__inference_flatten_4_layer_call_fn_139120956

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_139119643a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
яJ
г

F__inference_model_2_layer_call_and_return_conditional_losses_139120045

inputs
inputs_1.
conv2d_10_139119984:"
conv2d_10_139119986:	/
conv2d_11_139119989:"
conv2d_11_139119991:	/
conv2d_12_139119995:"
conv2d_12_139119997:	.
conv2d_13_139120000:@!
conv2d_13_139120002:@$
dense_12_139120005:@ 
dense_12_139120007:@$
dense_13_139120011:@  
dense_13_139120013: -
conv2d_14_139120016:@@!
conv2d_14_139120018:@&
dense_14_139120024:
 !
dense_14_139120026:	&
dense_15_139120029:
!
dense_15_139120031:	%
dense_16_139120034:	@ 
dense_16_139120036:@$
dense_17_139120039:@ 
dense_17_139120041:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_139119984conv2d_10_139119986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485Џ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_139119989conv2d_11_139119991*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509ћ
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139119443­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_139119995conv2d_12_139119997*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534Ў
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_139120000conv2d_13_139120002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_139120005dense_12_139120007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_139119582њ
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139119455Ё
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_139120011dense_13_139120013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_139119607Ќ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_139120016conv2d_14_139120018*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631ч
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_139119643х
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_139119651
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139119660
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_139120024dense_14_139120026*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_139119680Ђ
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_139120029dense_15_139120031*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_139119704Ё
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_139120034dense_16_139120036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_139119728Ё
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_139120039dense_17_139120041*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_139119744x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЬ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ		
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е

&__inference_internal_grad_fn_139121567
result_grads_0
result_grads_1
mul_conv2d_13_beta
mul_conv2d_13_biasadd
identity
mulMulmul_conv2d_13_betamul_conv2d_13_biasadd^result_grads_0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@q
mul_1Mulmul_conv2d_13_betamul_conv2d_13_biasadd*
T0*/
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*f
_input_shapesU
S:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:_ [
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:џџџџџџџџџ@
х

&__inference_internal_grad_fn_139121657
result_grads_0
result_grads_1
mul_dense_15_beta
mul_dense_15_biasadd
identityw
mulMulmul_dense_15_betamul_dense_15_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџh
mul_1Mulmul_dense_15_betamul_dense_15_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
ж

&__inference_internal_grad_fn_139121603
result_grads_0
result_grads_1
mul_dense_13_beta
mul_dense_13_biasadd
identityv
mulMulmul_dense_13_betamul_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ g
mul_1Mulmul_dense_13_betamul_dense_13_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :W S
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ 
Ў
}
&__inference_internal_grad_fn_139121477
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
ф

&__inference_internal_grad_fn_139121693
result_grads_0
result_grads_1
mul_conv2d_10_beta
mul_conv2d_10_biasadd
identity
mulMulmul_conv2d_10_betamul_conv2d_10_biasadd^result_grads_0*
T0*0
_output_shapes
:џџџџџџџџџ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		r
mul_1Mulmul_conv2d_10_betamul_conv2d_10_biasadd*
T0*0
_output_shapes
:џџџџџџџџџ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		"
identityIdentity:output:0*i
_input_shapesX
V:џџџџџџџџџ		:џџџџџџџџџ		: :џџџџџџџџџ		:` \
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:џџџџџџџџџ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:џџџџџџџџџ		
Э

,__inference_dense_12_layer_call_fn_139120879

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_139119582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
x
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139120986
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
:џџџџџџџџџ X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ:џџџџџџџџџ :R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/1
3
	
"__inference__traced_save_139122076
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ц

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*я	
valueх	Bт	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*
_input_shapesя
ь: :::::::@:@:@:@:@@:@:@ : :
 ::
::	@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:@: 
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
 :!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 
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
Ф

H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558

inputs9
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
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
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@Н
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119550*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г
ћ
G__inference_dense_16_layer_call_and_return_conditional_losses_139121067

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139121059*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж
Я
$__inference__wrapped_model_139119434
input_5
input_6K
0model_2_conv2d_10_conv2d_readvariableop_resource:@
1model_2_conv2d_10_biasadd_readvariableop_resource:	L
0model_2_conv2d_11_conv2d_readvariableop_resource:@
1model_2_conv2d_11_biasadd_readvariableop_resource:	L
0model_2_conv2d_12_conv2d_readvariableop_resource:@
1model_2_conv2d_12_biasadd_readvariableop_resource:	K
0model_2_conv2d_13_conv2d_readvariableop_resource:@?
1model_2_conv2d_13_biasadd_readvariableop_resource:@A
/model_2_dense_12_matmul_readvariableop_resource:@>
0model_2_dense_12_biasadd_readvariableop_resource:@A
/model_2_dense_13_matmul_readvariableop_resource:@ >
0model_2_dense_13_biasadd_readvariableop_resource: J
0model_2_conv2d_14_conv2d_readvariableop_resource:@@?
1model_2_conv2d_14_biasadd_readvariableop_resource:@C
/model_2_dense_14_matmul_readvariableop_resource:
 ?
0model_2_dense_14_biasadd_readvariableop_resource:	C
/model_2_dense_15_matmul_readvariableop_resource:
?
0model_2_dense_15_biasadd_readvariableop_resource:	B
/model_2_dense_16_matmul_readvariableop_resource:	@>
0model_2_dense_16_biasadd_readvariableop_resource:@A
/model_2_dense_17_matmul_readvariableop_resource:@>
0model_2_dense_17_biasadd_readvariableop_resource:
identityЂ(model_2/conv2d_10/BiasAdd/ReadVariableOpЂ'model_2/conv2d_10/Conv2D/ReadVariableOpЂ(model_2/conv2d_11/BiasAdd/ReadVariableOpЂ'model_2/conv2d_11/Conv2D/ReadVariableOpЂ(model_2/conv2d_12/BiasAdd/ReadVariableOpЂ'model_2/conv2d_12/Conv2D/ReadVariableOpЂ(model_2/conv2d_13/BiasAdd/ReadVariableOpЂ'model_2/conv2d_13/Conv2D/ReadVariableOpЂ(model_2/conv2d_14/BiasAdd/ReadVariableOpЂ'model_2/conv2d_14/Conv2D/ReadVariableOpЂ'model_2/dense_12/BiasAdd/ReadVariableOpЂ&model_2/dense_12/MatMul/ReadVariableOpЂ'model_2/dense_13/BiasAdd/ReadVariableOpЂ&model_2/dense_13/MatMul/ReadVariableOpЂ'model_2/dense_14/BiasAdd/ReadVariableOpЂ&model_2/dense_14/MatMul/ReadVariableOpЂ'model_2/dense_15/BiasAdd/ReadVariableOpЂ&model_2/dense_15/MatMul/ReadVariableOpЂ'model_2/dense_16/BiasAdd/ReadVariableOpЂ&model_2/dense_16/MatMul/ReadVariableOpЂ'model_2/dense_17/BiasAdd/ReadVariableOpЂ&model_2/dense_17/MatMul/ReadVariableOpЁ
'model_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0П
model_2/conv2d_10/Conv2DConv2Dinput_5/model_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides

(model_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
model_2/conv2d_10/BiasAddBiasAdd!model_2/conv2d_10/Conv2D:output:00model_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
model_2/conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_10/mulMulmodel_2/conv2d_10/beta:output:0"model_2/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		z
model_2/conv2d_10/SigmoidSigmoidmodel_2/conv2d_10/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
model_2/conv2d_10/mul_1Mul"model_2/conv2d_10/BiasAdd:output:0model_2/conv2d_10/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		~
model_2/conv2d_10/IdentityIdentitymodel_2/conv2d_10/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		ѕ
model_2/conv2d_10/IdentityN	IdentityNmodel_2/conv2d_10/mul_1:z:0"model_2/conv2d_10/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119286*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		Ђ
'model_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0м
model_2/conv2d_11/Conv2DConv2D$model_2/conv2d_10/IdentityN:output:0/model_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		*
paddingSAME*
strides

(model_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
model_2/conv2d_11/BiasAddBiasAdd!model_2/conv2d_11/Conv2D:output:00model_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ		[
model_2/conv2d_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_11/mulMulmodel_2/conv2d_11/beta:output:0"model_2/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ		z
model_2/conv2d_11/SigmoidSigmoidmodel_2/conv2d_11/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		
model_2/conv2d_11/mul_1Mul"model_2/conv2d_11/BiasAdd:output:0model_2/conv2d_11/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ		~
model_2/conv2d_11/IdentityIdentitymodel_2/conv2d_11/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџ		ѕ
model_2/conv2d_11/IdentityN	IdentityNmodel_2/conv2d_11/mul_1:z:0"model_2/conv2d_11/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119300*L
_output_shapes:
8:џџџџџџџџџ		:џџџџџџџџџ		О
model_2/max_pooling2d_4/MaxPoolMaxPool$model_2/conv2d_11/IdentityN:output:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
Ђ
'model_2/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0р
model_2/conv2d_12/Conv2DConv2D(model_2/max_pooling2d_4/MaxPool:output:0/model_2/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides

(model_2/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Д
model_2/conv2d_12/BiasAddBiasAdd!model_2/conv2d_12/Conv2D:output:00model_2/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ[
model_2/conv2d_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_12/mulMulmodel_2/conv2d_12/beta:output:0"model_2/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџz
model_2/conv2d_12/SigmoidSigmoidmodel_2/conv2d_12/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ
model_2/conv2d_12/mul_1Mul"model_2/conv2d_12/BiasAdd:output:0model_2/conv2d_12/Sigmoid:y:0*
T0*0
_output_shapes
:џџџџџџџџџ~
model_2/conv2d_12/IdentityIdentitymodel_2/conv2d_12/mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџѕ
model_2/conv2d_12/IdentityN	IdentityNmodel_2/conv2d_12/mul_1:z:0"model_2/conv2d_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119315*L
_output_shapes:
8:џџџџџџџџџ:џџџџџџџџџЁ
'model_2/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0л
model_2/conv2d_13/Conv2DConv2D$model_2/conv2d_12/IdentityN:output:0/model_2/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

(model_2/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
model_2/conv2d_13/BiasAddBiasAdd!model_2/conv2d_13/Conv2D:output:00model_2/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@[
model_2/conv2d_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_13/mulMulmodel_2/conv2d_13/beta:output:0"model_2/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@y
model_2/conv2d_13/SigmoidSigmoidmodel_2/conv2d_13/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
model_2/conv2d_13/mul_1Mul"model_2/conv2d_13/BiasAdd:output:0model_2/conv2d_13/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@}
model_2/conv2d_13/IdentityIdentitymodel_2/conv2d_13/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@ѓ
model_2/conv2d_13/IdentityN	IdentityNmodel_2/conv2d_13/mul_1:z:0"model_2/conv2d_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119329*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@
&model_2/dense_12/MatMul/ReadVariableOpReadVariableOp/model_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
model_2/dense_12/MatMulMatMulinput_6.model_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
'model_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model_2/dense_12/BiasAddBiasAdd!model_2/dense_12/MatMul:product:0/model_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
model_2/dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_12/mulMulmodel_2/dense_12/beta:output:0!model_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
model_2/dense_12/SigmoidSigmoidmodel_2/dense_12/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model_2/dense_12/mul_1Mul!model_2/dense_12/BiasAdd:output:0model_2/dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@s
model_2/dense_12/IdentityIdentitymodel_2/dense_12/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@р
model_2/dense_12/IdentityN	IdentityNmodel_2/dense_12/mul_1:z:0!model_2/dense_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119343*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@Н
model_2/max_pooling2d_5/MaxPoolMaxPool$model_2/conv2d_13/IdentityN:output:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

&model_2/dense_13/MatMul/ReadVariableOpReadVariableOp/model_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ј
model_2/dense_13/MatMulMatMul#model_2/dense_12/IdentityN:output:0.model_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'model_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
model_2/dense_13/BiasAddBiasAdd!model_2/dense_13/MatMul:product:0/model_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ Z
model_2/dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_13/mulMulmodel_2/dense_13/beta:output:0!model_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ o
model_2/dense_13/SigmoidSigmoidmodel_2/dense_13/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ 
model_2/dense_13/mul_1Mul!model_2/dense_13/BiasAdd:output:0model_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ s
model_2/dense_13/IdentityIdentitymodel_2/dense_13/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ р
model_2/dense_13/IdentityN	IdentityNmodel_2/dense_13/mul_1:z:0!model_2/dense_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119358*:
_output_shapes(
&:џџџџџџџџџ :џџџџџџџџџ  
'model_2/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0п
model_2/conv2d_14/Conv2DConv2D(model_2/max_pooling2d_5/MaxPool:output:0/model_2/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

(model_2/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
model_2/conv2d_14/BiasAddBiasAdd!model_2/conv2d_14/Conv2D:output:00model_2/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@[
model_2/conv2d_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_14/mulMulmodel_2/conv2d_14/beta:output:0"model_2/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@y
model_2/conv2d_14/SigmoidSigmoidmodel_2/conv2d_14/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@
model_2/conv2d_14/mul_1Mul"model_2/conv2d_14/BiasAdd:output:0model_2/conv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:џџџџџџџџџ@}
model_2/conv2d_14/IdentityIdentitymodel_2/conv2d_14/mul_1:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@ѓ
model_2/conv2d_14/IdentityN	IdentityNmodel_2/conv2d_14/mul_1:z:0"model_2/conv2d_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119372*J
_output_shapes8
6:џџџџџџџџџ@:џџџџџџџџџ@h
model_2/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
model_2/flatten_4/ReshapeReshape$model_2/conv2d_14/IdentityN:output:0 model_2/flatten_4/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџh
model_2/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    
model_2/flatten_5/ReshapeReshape#model_2/dense_13/IdentityN:output:0 model_2/flatten_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :и
model_2/concatenate_2/concatConcatV2"model_2/flatten_4/Reshape:output:0"model_2/flatten_5/Reshape:output:0*model_2/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ 
&model_2/dense_14/MatMul/ReadVariableOpReadVariableOp/model_2_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0Ћ
model_2/dense_14/MatMulMatMul%model_2/concatenate_2/concat:output:0.model_2/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
'model_2/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Њ
model_2/dense_14/BiasAddBiasAdd!model_2/dense_14/MatMul:product:0/model_2/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџZ
model_2/dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_14/mulMulmodel_2/dense_14/beta:output:0!model_2/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџp
model_2/dense_14/SigmoidSigmoidmodel_2/dense_14/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
model_2/dense_14/mul_1Mul!model_2/dense_14/BiasAdd:output:0model_2/dense_14/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџt
model_2/dense_14/IdentityIdentitymodel_2/dense_14/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџт
model_2/dense_14/IdentityN	IdentityNmodel_2/dense_14/mul_1:z:0!model_2/dense_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119392*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
&model_2/dense_15/MatMul/ReadVariableOpReadVariableOp/model_2_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Љ
model_2/dense_15/MatMulMatMul#model_2/dense_14/IdentityN:output:0.model_2/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
'model_2/dense_15/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Њ
model_2/dense_15/BiasAddBiasAdd!model_2/dense_15/MatMul:product:0/model_2/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџZ
model_2/dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_15/mulMulmodel_2/dense_15/beta:output:0!model_2/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџp
model_2/dense_15/SigmoidSigmoidmodel_2/dense_15/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
model_2/dense_15/mul_1Mul!model_2/dense_15/BiasAdd:output:0model_2/dense_15/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџt
model_2/dense_15/IdentityIdentitymodel_2/dense_15/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџт
model_2/dense_15/IdentityN	IdentityNmodel_2/dense_15/mul_1:z:0!model_2/dense_15/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119406*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ
&model_2/dense_16/MatMul/ReadVariableOpReadVariableOp/model_2_dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ј
model_2/dense_16/MatMulMatMul#model_2/dense_15/IdentityN:output:0.model_2/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
'model_2/dense_16/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model_2/dense_16/BiasAddBiasAdd!model_2/dense_16/MatMul:product:0/model_2/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Z
model_2/dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_16/mulMulmodel_2/dense_16/beta:output:0!model_2/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
model_2/dense_16/SigmoidSigmoidmodel_2/dense_16/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model_2/dense_16/mul_1Mul!model_2/dense_16/BiasAdd:output:0model_2/dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@s
model_2/dense_16/IdentityIdentitymodel_2/dense_16/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@р
model_2/dense_16/IdentityN	IdentityNmodel_2/dense_16/mul_1:z:0!model_2/dense_16/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119420*:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@
&model_2/dense_17/MatMul/ReadVariableOpReadVariableOp/model_2_dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
model_2/dense_17/MatMulMatMul#model_2/dense_16/IdentityN:output:0.model_2/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model_2/dense_17/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model_2/dense_17/BiasAddBiasAdd!model_2/dense_17/MatMul:product:0/model_2/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
IdentityIdentity!model_2/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџс
NoOpNoOp)^model_2/conv2d_10/BiasAdd/ReadVariableOp(^model_2/conv2d_10/Conv2D/ReadVariableOp)^model_2/conv2d_11/BiasAdd/ReadVariableOp(^model_2/conv2d_11/Conv2D/ReadVariableOp)^model_2/conv2d_12/BiasAdd/ReadVariableOp(^model_2/conv2d_12/Conv2D/ReadVariableOp)^model_2/conv2d_13/BiasAdd/ReadVariableOp(^model_2/conv2d_13/Conv2D/ReadVariableOp)^model_2/conv2d_14/BiasAdd/ReadVariableOp(^model_2/conv2d_14/Conv2D/ReadVariableOp(^model_2/dense_12/BiasAdd/ReadVariableOp'^model_2/dense_12/MatMul/ReadVariableOp(^model_2/dense_13/BiasAdd/ReadVariableOp'^model_2/dense_13/MatMul/ReadVariableOp(^model_2/dense_14/BiasAdd/ReadVariableOp'^model_2/dense_14/MatMul/ReadVariableOp(^model_2/dense_15/BiasAdd/ReadVariableOp'^model_2/dense_15/MatMul/ReadVariableOp(^model_2/dense_16/BiasAdd/ReadVariableOp'^model_2/dense_16/MatMul/ReadVariableOp(^model_2/dense_17/BiasAdd/ReadVariableOp'^model_2/dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:џџџџџџџџџ		:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : 2T
(model_2/conv2d_10/BiasAdd/ReadVariableOp(model_2/conv2d_10/BiasAdd/ReadVariableOp2R
'model_2/conv2d_10/Conv2D/ReadVariableOp'model_2/conv2d_10/Conv2D/ReadVariableOp2T
(model_2/conv2d_11/BiasAdd/ReadVariableOp(model_2/conv2d_11/BiasAdd/ReadVariableOp2R
'model_2/conv2d_11/Conv2D/ReadVariableOp'model_2/conv2d_11/Conv2D/ReadVariableOp2T
(model_2/conv2d_12/BiasAdd/ReadVariableOp(model_2/conv2d_12/BiasAdd/ReadVariableOp2R
'model_2/conv2d_12/Conv2D/ReadVariableOp'model_2/conv2d_12/Conv2D/ReadVariableOp2T
(model_2/conv2d_13/BiasAdd/ReadVariableOp(model_2/conv2d_13/BiasAdd/ReadVariableOp2R
'model_2/conv2d_13/Conv2D/ReadVariableOp'model_2/conv2d_13/Conv2D/ReadVariableOp2T
(model_2/conv2d_14/BiasAdd/ReadVariableOp(model_2/conv2d_14/BiasAdd/ReadVariableOp2R
'model_2/conv2d_14/Conv2D/ReadVariableOp'model_2/conv2d_14/Conv2D/ReadVariableOp2R
'model_2/dense_12/BiasAdd/ReadVariableOp'model_2/dense_12/BiasAdd/ReadVariableOp2P
&model_2/dense_12/MatMul/ReadVariableOp&model_2/dense_12/MatMul/ReadVariableOp2R
'model_2/dense_13/BiasAdd/ReadVariableOp'model_2/dense_13/BiasAdd/ReadVariableOp2P
&model_2/dense_13/MatMul/ReadVariableOp&model_2/dense_13/MatMul/ReadVariableOp2R
'model_2/dense_14/BiasAdd/ReadVariableOp'model_2/dense_14/BiasAdd/ReadVariableOp2P
&model_2/dense_14/MatMul/ReadVariableOp&model_2/dense_14/MatMul/ReadVariableOp2R
'model_2/dense_15/BiasAdd/ReadVariableOp'model_2/dense_15/BiasAdd/ReadVariableOp2P
&model_2/dense_15/MatMul/ReadVariableOp&model_2/dense_15/MatMul/ReadVariableOp2R
'model_2/dense_16/BiasAdd/ReadVariableOp'model_2/dense_16/BiasAdd/ReadVariableOp2P
&model_2/dense_16/MatMul/ReadVariableOp&model_2/dense_16/MatMul/ReadVariableOp2R
'model_2/dense_17/BiasAdd/ReadVariableOp'model_2/dense_17/BiasAdd/ReadVariableOp2P
&model_2/dense_17/MatMul/ReadVariableOp&model_2/dense_17/MatMul/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
Р
§
G__inference_dense_15_layer_call_and_return_conditional_losses_139119704

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџЏ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-139119696*<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х

&__inference_internal_grad_fn_139121837
result_grads_0
result_grads_1
mul_dense_15_beta
mul_dense_15_biasadd
identityw
mulMulmul_dense_15_betamul_dense_15_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџh
mul_1Mulmul_dense_15_betamul_dense_15_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
Л
]
1__inference_concatenate_2_layer_call_fn_139120979
inputs_0
inputs_1
identityЪ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139119660a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ:џџџџџџџџџ :R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs/1
х

&__inference_internal_grad_fn_139121819
result_grads_0
result_grads_1
mul_dense_14_beta
mul_dense_14_biasadd
identityw
mulMulmul_dense_14_betamul_dense_14_biasadd^result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџh
mul_1Mulmul_dense_14_betamul_dense_14_biasadd*
T0*(
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:џџџџџџџџџU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:џџџџџџџџџZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*Q
_input_shapes@
>:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ
Ъ
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_139120962

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ж

&__inference_internal_grad_fn_139121765
result_grads_0
result_grads_1
mul_dense_12_beta
mul_dense_12_biasadd
identityv
mulMulmul_dense_12_betamul_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
mul_1Mulmul_dense_12_betamul_dense_12_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@

}
&__inference_internal_grad_fn_139121945
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*N
_input_shapes=
;:џџџџџџџџџ@:џџџџџџџџџ@: :џџџџџџџџџ@:W S
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@B
&__inference_internal_grad_fn_139121153CustomGradient-139119286B
&__inference_internal_grad_fn_139121171CustomGradient-139119300B
&__inference_internal_grad_fn_139121189CustomGradient-139119315B
&__inference_internal_grad_fn_139121207CustomGradient-139119329B
&__inference_internal_grad_fn_139121225CustomGradient-139119343B
&__inference_internal_grad_fn_139121243CustomGradient-139119358B
&__inference_internal_grad_fn_139121261CustomGradient-139119372B
&__inference_internal_grad_fn_139121279CustomGradient-139119392B
&__inference_internal_grad_fn_139121297CustomGradient-139119406B
&__inference_internal_grad_fn_139121315CustomGradient-139119420B
&__inference_internal_grad_fn_139121333CustomGradient-139119477B
&__inference_internal_grad_fn_139121351CustomGradient-139119501B
&__inference_internal_grad_fn_139121369CustomGradient-139119526B
&__inference_internal_grad_fn_139121387CustomGradient-139119550B
&__inference_internal_grad_fn_139121405CustomGradient-139119574B
&__inference_internal_grad_fn_139121423CustomGradient-139119599B
&__inference_internal_grad_fn_139121441CustomGradient-139119623B
&__inference_internal_grad_fn_139121459CustomGradient-139119672B
&__inference_internal_grad_fn_139121477CustomGradient-139119696B
&__inference_internal_grad_fn_139121495CustomGradient-139119720B
&__inference_internal_grad_fn_139121513CustomGradient-139120383B
&__inference_internal_grad_fn_139121531CustomGradient-139120397B
&__inference_internal_grad_fn_139121549CustomGradient-139120412B
&__inference_internal_grad_fn_139121567CustomGradient-139120426B
&__inference_internal_grad_fn_139121585CustomGradient-139120440B
&__inference_internal_grad_fn_139121603CustomGradient-139120455B
&__inference_internal_grad_fn_139121621CustomGradient-139120469B
&__inference_internal_grad_fn_139121639CustomGradient-139120489B
&__inference_internal_grad_fn_139121657CustomGradient-139120503B
&__inference_internal_grad_fn_139121675CustomGradient-139120517B
&__inference_internal_grad_fn_139121693CustomGradient-139120542B
&__inference_internal_grad_fn_139121711CustomGradient-139120556B
&__inference_internal_grad_fn_139121729CustomGradient-139120571B
&__inference_internal_grad_fn_139121747CustomGradient-139120585B
&__inference_internal_grad_fn_139121765CustomGradient-139120599B
&__inference_internal_grad_fn_139121783CustomGradient-139120614B
&__inference_internal_grad_fn_139121801CustomGradient-139120628B
&__inference_internal_grad_fn_139121819CustomGradient-139120648B
&__inference_internal_grad_fn_139121837CustomGradient-139120662B
&__inference_internal_grad_fn_139121855CustomGradient-139120676B
&__inference_internal_grad_fn_139121873CustomGradient-139120761B
&__inference_internal_grad_fn_139121891CustomGradient-139120788B
&__inference_internal_grad_fn_139121909CustomGradient-139120825B
&__inference_internal_grad_fn_139121927CustomGradient-139120852B
&__inference_internal_grad_fn_139121945CustomGradient-139120889B
&__inference_internal_grad_fn_139121963CustomGradient-139120916B
&__inference_internal_grad_fn_139121981CustomGradient-139120943B
&__inference_internal_grad_fn_139121999CustomGradient-139121005B
&__inference_internal_grad_fn_139122017CustomGradient-139121032B
&__inference_internal_grad_fn_139122035CustomGradient-139121059"лL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*№
serving_defaultм
C
input_58
serving_default_input_5:0џџџџџџџџџ		
;
input_60
serving_default_input_6:0џџџџџџџџџ<
dense_170
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:С

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
р

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
р

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
Ъ
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
р

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
р

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
Ъ
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
р

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
р

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
р

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
Ъ
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
Ъ
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
Ъ
#y_self_saveable_object_factories
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
щ
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
щ
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
щ
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
щ
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
 regularization_losses
Ё	keras_api
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
-
Єserving_default"
signature_map
 "
trackable_dict_wrapper
Ю
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
14
15
16
17
18
19
20
21"
trackable_list_wrapper
Ю
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
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
њ2ї
+__inference_model_2_layer_call_fn_139119798
+__inference_model_2_layer_call_fn_139120322
+__inference_model_2_layer_call_fn_139120372
+__inference_model_2_layer_call_fn_139120142Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
F__inference_model_2_layer_call_and_return_conditional_losses_139120531
F__inference_model_2_layer_call_and_return_conditional_losses_139120690
F__inference_model_2_layer_call_and_return_conditional_losses_139120207
F__inference_model_2_layer_call_and_return_conditional_losses_139120272Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
иBе
$__inference__wrapped_model_139119434input_5input_6"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_dict_wrapper
+:)2conv2d_10/kernel
:2conv2d_10/bias
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
В
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
з2д
-__inference_conv2d_10_layer_call_fn_139120751Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_conv2d_10_layer_call_and_return_conditional_losses_139120769Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
,:*2conv2d_11/kernel
:2conv2d_11/bias
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
В
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
з2д
-__inference_conv2d_11_layer_call_fn_139120778Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_conv2d_11_layer_call_and_return_conditional_losses_139120796Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
н2к
3__inference_max_pooling2d_4_layer_call_fn_139120801Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139120806Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
,:*2conv2d_12/kernel
:2conv2d_12/bias
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
В
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
з2д
-__inference_conv2d_12_layer_call_fn_139120815Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_conv2d_12_layer_call_and_return_conditional_losses_139120833Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
+:)@2conv2d_13/kernel
:@2conv2d_13/bias
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
В
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
з2д
-__inference_conv2d_13_layer_call_fn_139120842Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_conv2d_13_layer_call_and_return_conditional_losses_139120860Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
В
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
н2к
3__inference_max_pooling2d_5_layer_call_fn_139120865Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139120870Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!:@2dense_12/kernel
:@2dense_12/bias
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
В
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
ж2г
,__inference_dense_12_layer_call_fn_139120879Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_12_layer_call_and_return_conditional_losses_139120897Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
*:(@@2conv2d_14/kernel
:@2conv2d_14/bias
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
В
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
з2д
-__inference_conv2d_14_layer_call_fn_139120906Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_conv2d_14_layer_call_and_return_conditional_losses_139120924Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!:@ 2dense_13/kernel
: 2dense_13/bias
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
В
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ж2г
,__inference_dense_13_layer_call_fn_139120933Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_13_layer_call_and_return_conditional_losses_139120951Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
зnon_trainable_variables
иlayers
йmetrics
 кlayer_regularization_losses
лlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
з2д
-__inference_flatten_4_layer_call_fn_139120956Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_flatten_4_layer_call_and_return_conditional_losses_139120962Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
з2д
-__inference_flatten_5_layer_call_fn_139120967Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_flatten_5_layer_call_and_return_conditional_losses_139120973Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
л2и
1__inference_concatenate_2_layer_call_fn_139120979Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139120986Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
#:!
 2dense_14/kernel
:2dense_14/bias
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ж2г
,__inference_dense_14_layer_call_fn_139120995Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_14_layer_call_and_return_conditional_losses_139121013Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
#:!
2dense_15/kernel
:2dense_15/bias
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ж2г
,__inference_dense_15_layer_call_fn_139121022Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_15_layer_call_and_return_conditional_losses_139121040Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
": 	@2dense_16/kernel
:@2dense_16/bias
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
№non_trainable_variables
ёlayers
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ж2г
,__inference_dense_16_layer_call_fn_139121049Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_16_layer_call_and_return_conditional_losses_139121067Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!:@2dense_17/kernel
:2dense_17/bias
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѕnon_trainable_variables
іlayers
їmetrics
 јlayer_regularization_losses
љlayer_metrics
	variables
trainable_variables
 regularization_losses
Ђ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
ж2г
,__inference_dense_17_layer_call_fn_139121076Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_17_layer_call_and_return_conditional_losses_139121086Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
еBв
'__inference_signature_wrapper_139120742input_5input_6"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
І
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
Bb@
model_2/conv2d_10/beta:0$__inference__wrapped_model_139119434
EbC
model_2/conv2d_10/BiasAdd:0$__inference__wrapped_model_139119434
Bb@
model_2/conv2d_11/beta:0$__inference__wrapped_model_139119434
EbC
model_2/conv2d_11/BiasAdd:0$__inference__wrapped_model_139119434
Bb@
model_2/conv2d_12/beta:0$__inference__wrapped_model_139119434
EbC
model_2/conv2d_12/BiasAdd:0$__inference__wrapped_model_139119434
Bb@
model_2/conv2d_13/beta:0$__inference__wrapped_model_139119434
EbC
model_2/conv2d_13/BiasAdd:0$__inference__wrapped_model_139119434
Ab?
model_2/dense_12/beta:0$__inference__wrapped_model_139119434
DbB
model_2/dense_12/BiasAdd:0$__inference__wrapped_model_139119434
Ab?
model_2/dense_13/beta:0$__inference__wrapped_model_139119434
DbB
model_2/dense_13/BiasAdd:0$__inference__wrapped_model_139119434
Bb@
model_2/conv2d_14/beta:0$__inference__wrapped_model_139119434
EbC
model_2/conv2d_14/BiasAdd:0$__inference__wrapped_model_139119434
Ab?
model_2/dense_14/beta:0$__inference__wrapped_model_139119434
DbB
model_2/dense_14/BiasAdd:0$__inference__wrapped_model_139119434
Ab?
model_2/dense_15/beta:0$__inference__wrapped_model_139119434
DbB
model_2/dense_15/BiasAdd:0$__inference__wrapped_model_139119434
Ab?
model_2/dense_16/beta:0$__inference__wrapped_model_139119434
DbB
model_2/dense_16/BiasAdd:0$__inference__wrapped_model_139119434
TbR
beta:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485
WbU
	BiasAdd:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_139119485
TbR
beta:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509
WbU
	BiasAdd:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_139119509
TbR
beta:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534
WbU
	BiasAdd:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_139119534
TbR
beta:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558
WbU
	BiasAdd:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_139119558
SbQ
beta:0G__inference_dense_12_layer_call_and_return_conditional_losses_139119582
VbT
	BiasAdd:0G__inference_dense_12_layer_call_and_return_conditional_losses_139119582
SbQ
beta:0G__inference_dense_13_layer_call_and_return_conditional_losses_139119607
VbT
	BiasAdd:0G__inference_dense_13_layer_call_and_return_conditional_losses_139119607
TbR
beta:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631
WbU
	BiasAdd:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_139119631
SbQ
beta:0G__inference_dense_14_layer_call_and_return_conditional_losses_139119680
VbT
	BiasAdd:0G__inference_dense_14_layer_call_and_return_conditional_losses_139119680
SbQ
beta:0G__inference_dense_15_layer_call_and_return_conditional_losses_139119704
VbT
	BiasAdd:0G__inference_dense_15_layer_call_and_return_conditional_losses_139119704
SbQ
beta:0G__inference_dense_16_layer_call_and_return_conditional_losses_139119728
VbT
	BiasAdd:0G__inference_dense_16_layer_call_and_return_conditional_losses_139119728
\bZ
conv2d_10/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
_b]
conv2d_10/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
\bZ
conv2d_11/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
_b]
conv2d_11/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
\bZ
conv2d_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
_b]
conv2d_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
\bZ
conv2d_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
_b]
conv2d_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
[bY
dense_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
^b\
dense_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
[bY
dense_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
^b\
dense_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
\bZ
conv2d_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
_b]
conv2d_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
[bY
dense_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
^b\
dense_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
[bY
dense_15/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
^b\
dense_15/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
[bY
dense_16/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
^b\
dense_16/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120531
\bZ
conv2d_10/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
_b]
conv2d_10/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
\bZ
conv2d_11/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
_b]
conv2d_11/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
\bZ
conv2d_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
_b]
conv2d_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
\bZ
conv2d_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
_b]
conv2d_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
[bY
dense_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
^b\
dense_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
[bY
dense_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
^b\
dense_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
\bZ
conv2d_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
_b]
conv2d_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
[bY
dense_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
^b\
dense_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
[bY
dense_15/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
^b\
dense_15/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
[bY
dense_16/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
^b\
dense_16/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_139120690
TbR
beta:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_139120769
WbU
	BiasAdd:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_139120769
TbR
beta:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_139120796
WbU
	BiasAdd:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_139120796
TbR
beta:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_139120833
WbU
	BiasAdd:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_139120833
TbR
beta:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_139120860
WbU
	BiasAdd:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_139120860
SbQ
beta:0G__inference_dense_12_layer_call_and_return_conditional_losses_139120897
VbT
	BiasAdd:0G__inference_dense_12_layer_call_and_return_conditional_losses_139120897
TbR
beta:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_139120924
WbU
	BiasAdd:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_139120924
SbQ
beta:0G__inference_dense_13_layer_call_and_return_conditional_losses_139120951
VbT
	BiasAdd:0G__inference_dense_13_layer_call_and_return_conditional_losses_139120951
SbQ
beta:0G__inference_dense_14_layer_call_and_return_conditional_losses_139121013
VbT
	BiasAdd:0G__inference_dense_14_layer_call_and_return_conditional_losses_139121013
SbQ
beta:0G__inference_dense_15_layer_call_and_return_conditional_losses_139121040
VbT
	BiasAdd:0G__inference_dense_15_layer_call_and_return_conditional_losses_139121040
SbQ
beta:0G__inference_dense_16_layer_call_and_return_conditional_losses_139121067
VbT
	BiasAdd:0G__inference_dense_16_layer_call_and_return_conditional_losses_139121067р
$__inference__wrapped_model_139119434З&'67?@PQbcYZ`Ђ]
VЂS
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
Њ "3Њ0
.
dense_17"
dense_17џџџџџџџџџж
L__inference_concatenate_2_layer_call_and_return_conditional_losses_139120986[ЂX
QЂN
LI
# 
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџ 
 ­
1__inference_concatenate_2_layer_call_fn_139120979x[ЂX
QЂN
LI
# 
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ 
Њ "џџџџџџџџџ Й
H__inference_conv2d_10_layer_call_and_return_conditional_losses_139120769m7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		
Њ ".Ђ+
$!
0џџџџџџџџџ		
 
-__inference_conv2d_10_layer_call_fn_139120751`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		
Њ "!џџџџџџџџџ		К
H__inference_conv2d_11_layer_call_and_return_conditional_losses_139120796n&'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ		
Њ ".Ђ+
$!
0џџџџџџџџџ		
 
-__inference_conv2d_11_layer_call_fn_139120778a&'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ		
Њ "!џџџџџџџџџ		К
H__inference_conv2d_12_layer_call_and_return_conditional_losses_139120833n678Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
-__inference_conv2d_12_layer_call_fn_139120815a678Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЙ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_139120860m?@8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
-__inference_conv2d_13_layer_call_fn_139120842`?@8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ " џџџџџџџџџ@И
H__inference_conv2d_14_layer_call_and_return_conditional_losses_139120924lYZ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
-__inference_conv2d_14_layer_call_fn_139120906_YZ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@Ї
G__inference_dense_12_layer_call_and_return_conditional_losses_139120897\PQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 
,__inference_dense_12_layer_call_fn_139120879OPQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ї
G__inference_dense_13_layer_call_and_return_conditional_losses_139120951\bc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 
,__inference_dense_13_layer_call_fn_139120933Obc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ Ћ
G__inference_dense_14_layer_call_and_return_conditional_losses_139121013`0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_dense_14_layer_call_fn_139120995S0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЋ
G__inference_dense_15_layer_call_and_return_conditional_losses_139121040`0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_dense_15_layer_call_fn_139121022S0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЊ
G__inference_dense_16_layer_call_and_return_conditional_losses_139121067_0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 
,__inference_dense_16_layer_call_fn_139121049R0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Љ
G__inference_dense_17_layer_call_and_return_conditional_losses_139121086^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_dense_17_layer_call_fn_139121076Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ­
H__inference_flatten_4_layer_call_and_return_conditional_losses_139120962a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 
-__inference_flatten_4_layer_call_fn_139120956T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЄ
H__inference_flatten_5_layer_call_and_return_conditional_losses_139120973X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 |
-__inference_flatten_5_layer_call_fn_139120967K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ й
&__inference_internal_grad_fn_139121153ЎњћwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121171Ўќ§wЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121189ЎўџwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ
1.
result_grads_1џџџџџџџџџ
Њ "-*

 
$!
1џџџџџџџџџж
&__inference_internal_grad_fn_139121207ЋuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121225eЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121243eЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ 
(%
result_grads_1џџџџџџџџџ 
Њ "$!

 

1џџџџџџџџџ ж
&__inference_internal_grad_fn_139121261ЋuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@С
&__inference_internal_grad_fn_139121279gЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџС
&__inference_internal_grad_fn_139121297gЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџО
&__inference_internal_grad_fn_139121315eЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@й
&__inference_internal_grad_fn_139121333ЎwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121351ЎwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121369ЎwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ
1.
result_grads_1џџџџџџџџџ
Њ "-*

 
$!
1џџџџџџџџџж
&__inference_internal_grad_fn_139121387ЋuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121405eЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121423eЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ 
(%
result_grads_1џџџџџџџџџ 
Њ "$!

 

1џџџџџџџџџ ж
&__inference_internal_grad_fn_139121441ЋuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@С
&__inference_internal_grad_fn_139121459gЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџС
&__inference_internal_grad_fn_139121477gЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџО
&__inference_internal_grad_fn_139121495 ЁeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@й
&__inference_internal_grad_fn_139121513ЎЂЃwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121531ЎЄЅwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121549ЎІЇwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ
1.
result_grads_1џџџџџџџџџ
Њ "-*

 
$!
1џџџџџџџџџж
&__inference_internal_grad_fn_139121567ЋЈЉuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121585ЊЋeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121603Ќ­eЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ 
(%
result_grads_1џџџџџџџџџ 
Њ "$!

 

1џџџџџџџџџ ж
&__inference_internal_grad_fn_139121621ЋЎЏuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@С
&__inference_internal_grad_fn_139121639АБgЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџС
&__inference_internal_grad_fn_139121657ВГgЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџО
&__inference_internal_grad_fn_139121675ДЕeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@й
&__inference_internal_grad_fn_139121693ЎЖЗwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121711ЎИЙwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121729ЎКЛwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ
1.
result_grads_1џџџџџџџџџ
Њ "-*

 
$!
1џџџџџџџџџж
&__inference_internal_grad_fn_139121747ЋМНuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121765ОПeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121783РСeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ 
(%
result_grads_1џџџџџџџџџ 
Њ "$!

 

1џџџџџџџџџ ж
&__inference_internal_grad_fn_139121801ЋТУuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@С
&__inference_internal_grad_fn_139121819ФХgЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџС
&__inference_internal_grad_fn_139121837ЦЧgЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџО
&__inference_internal_grad_fn_139121855ШЩeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@й
&__inference_internal_grad_fn_139121873ЎЪЫwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121891ЎЬЭwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ		
1.
result_grads_1џџџџџџџџџ		
Њ "-*

 
$!
1џџџџџџџџџ		й
&__inference_internal_grad_fn_139121909ЎЮЯwЂt
mЂj

 
1.
result_grads_0џџџџџџџџџ
1.
result_grads_1џџџџџџџџџ
Њ "-*

 
$!
1џџџџџџџџџж
&__inference_internal_grad_fn_139121927ЋабuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121945вгeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@ж
&__inference_internal_grad_fn_139121963ЋдеuЂr
kЂh

 
0-
result_grads_0џџџџџџџџџ@
0-
result_grads_1џџџџџџџџџ@
Њ ",)

 
# 
1џџџџџџџџџ@О
&__inference_internal_grad_fn_139121981жзeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ 
(%
result_grads_1џџџџџџџџџ 
Њ "$!

 

1џџџџџџџџџ С
&__inference_internal_grad_fn_139121999ийgЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџС
&__inference_internal_grad_fn_139122017клgЂd
]ЂZ

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
Њ "%"

 

1џџџџџџџџџО
&__inference_internal_grad_fn_139122035мнeЂb
[ЂX

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@
Њ "$!

 

1џџџџџџџџџ@ё
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_139120806RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Щ
3__inference_max_pooling2d_4_layer_call_fn_139120801RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџё
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_139120870RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Щ
3__inference_max_pooling2d_5_layer_call_fn_139120865RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџќ
F__inference_model_2_layer_call_and_return_conditional_losses_139120207Б&'67?@PQbcYZhЂe
^Ђ[
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 ќ
F__inference_model_2_layer_call_and_return_conditional_losses_139120272Б&'67?@PQbcYZhЂe
^Ђ[
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 ў
F__inference_model_2_layer_call_and_return_conditional_losses_139120531Г&'67?@PQbcYZjЂg
`Ђ]
SP
*'
inputs/0џџџџџџџџџ		
"
inputs/1џџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 ў
F__inference_model_2_layer_call_and_return_conditional_losses_139120690Г&'67?@PQbcYZjЂg
`Ђ]
SP
*'
inputs/0џџџџџџџџџ		
"
inputs/1џџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 д
+__inference_model_2_layer_call_fn_139119798Є&'67?@PQbcYZhЂe
^Ђ[
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
p 

 
Њ "џџџџџџџџџд
+__inference_model_2_layer_call_fn_139120142Є&'67?@PQbcYZhЂe
^Ђ[
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
p

 
Њ "џџџџџџџџџж
+__inference_model_2_layer_call_fn_139120322І&'67?@PQbcYZjЂg
`Ђ]
SP
*'
inputs/0џџџџџџџџџ		
"
inputs/1џџџџџџџџџ
p 

 
Њ "џџџџџџџџџж
+__inference_model_2_layer_call_fn_139120372І&'67?@PQbcYZjЂg
`Ђ]
SP
*'
inputs/0џџџџџџџџџ		
"
inputs/1џџџџџџџџџ
p

 
Њ "џџџџџџџџџє
'__inference_signature_wrapper_139120742Ш&'67?@PQbcYZqЂn
Ђ 
gЊd
4
input_5)&
input_5џџџџџџџџџ		
,
input_6!
input_6џџџџџџџџџ"3Њ0
.
dense_17"
dense_17џџџџџџџџџ