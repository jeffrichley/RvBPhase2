ХІ
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68љ
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
щ
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
GPU2*0,1J 8 */
f*R(
&__inference_signature_wrapper_88346382
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
к
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
GPU2*0,1J 8 **
f%R#
!__inference__traced_save_88347716

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
GPU2*0,1J 8 *-
f(R&
$__inference__traced_restore_88347792ге
Я

G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174

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
:џџџџџџџџџО
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345166*L
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
Ѕ
|
%__inference_internal_grad_fn_88347009
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

|
%__inference_internal_grad_fn_88347585
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
Ы

G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125

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
:џџџџџџџџџ		О
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345117*L
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
Щ	
ї
F__inference_dense_17_layer_call_and_return_conditional_losses_88345384

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
в

+__inference_dense_14_layer_call_fn_88346635

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallс
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_88345320p
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
ВJ
М

E__inference_model_2_layer_call_and_return_conditional_losses_88345685

inputs
inputs_1-
conv2d_10_88345624:!
conv2d_10_88345626:	.
conv2d_11_88345629:!
conv2d_11_88345631:	.
conv2d_12_88345635:!
conv2d_12_88345637:	-
conv2d_13_88345640:@ 
conv2d_13_88345642:@#
dense_12_88345645:@
dense_12_88345647:@#
dense_13_88345651:@ 
dense_13_88345653: ,
conv2d_14_88345656:@@ 
conv2d_14_88345658:@%
dense_14_88345664:
  
dense_14_88345666:	%
dense_15_88345669:
 
dense_15_88345671:	$
dense_16_88345674:	@
dense_16_88345676:@#
dense_17_88345679:@
dense_17_88345681:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_88345624conv2d_10_88345626*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125Ќ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_88345629conv2d_11_88345631*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149њ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88345083Њ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_88345635conv2d_12_88345637*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174Ћ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_88345640conv2d_13_88345642*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198§
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_88345645dense_12_88345647*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_88345222љ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88345095
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_88345651dense_13_88345653*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_88345247Љ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_88345656conv2d_14_88345658*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271ц
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_88345283ф
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_88345291
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
GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88345300
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_88345664dense_14_88345666*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_88345320
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_88345669dense_15_88345671*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_88345344
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_88345674dense_16_88345676*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_88345368
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_88345679dense_17_88345681*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_88345384x
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

 
%__inference_internal_grad_fn_88346847
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
­
|
%__inference_internal_grad_fn_88347099
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
Р
u
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88345300

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
Я

G__inference_conv2d_12_layer_call_and_return_conditional_losses_88346473

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
:џџџџџџџџџО
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346465*L
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

|
%__inference_internal_grad_fn_88347675
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

|
%__inference_internal_grad_fn_88347027
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

|
%__inference_internal_grad_fn_88347045
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
С
N
2__inference_max_pooling2d_5_layer_call_fn_88346505

inputs
identityр
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88345095
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
Ѕ
|
%__inference_internal_grad_fn_88346973
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
Ы

+__inference_dense_17_layer_call_fn_88346716

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallр
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_88345384o
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
Ы

+__inference_dense_12_layer_call_fn_88346519

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallр
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_88345222o
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
Б
њ
F__inference_dense_16_layer_call_and_return_conditional_losses_88346707

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
:џџџџџџџџџ@Ќ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346699*:
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
О

G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271

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
:џџџџџџџџџ@М
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345263*J
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
д

%__inference_internal_grad_fn_88347261
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

 
%__inference_internal_grad_fn_88346811
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
О
ќ
F__inference_dense_14_layer_call_and_return_conditional_losses_88346653

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
:џџџџџџџџџЎ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346645*<
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

i
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88346446

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
у

%__inference_internal_grad_fn_88347189
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
­
љ
F__inference_dense_13_layer_call_and_return_conditional_losses_88345247

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
:џџџџџџџџџ Ќ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345239*:
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
е

%__inference_internal_grad_fn_88347405
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
в

+__inference_dense_15_layer_call_fn_88346662

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallс
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_88345344p
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
3
	
!__inference__traced_save_88347716
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
е

%__inference_internal_grad_fn_88347495
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
З
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_88345291

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
Ѕ
|
%__inference_internal_grad_fn_88347513
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

|
%__inference_internal_grad_fn_88347621
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
Ј
H
,__inference_flatten_5_layer_call_fn_88346607

inputs
identityЗ
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_88345291`
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
д

%__inference_internal_grad_fn_88347441
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
Ы

G__inference_conv2d_10_layer_call_and_return_conditional_losses_88346409

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
:џџџџџџџџџ		О
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346401*L
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
е

%__inference_internal_grad_fn_88347315
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
є

*__inference_model_2_layer_call_fn_88345962
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
identityЂStatefulPartitionedCallћ
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
GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_88345391o
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
Ш
w
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88346626
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
О
ќ
F__inference_dense_15_layer_call_and_return_conditional_losses_88346680

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
:џџџџџџџџџЎ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346672*<
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
ф

%__inference_internal_grad_fn_88347459
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

i
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88345095

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
в

E__inference_model_2_layer_call_and_return_conditional_losses_88346330
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
:џџџџџџџџџ		м
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346182*L
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
:џџџџџџџџџ		м
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346196*L
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
:џџџџџџџџџм
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346211*L
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
:џџџџџџџџџ@к
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346225*J
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
:џџџџџџџџџ@Ч
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346239*:
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
:џџџџџџџџџ Ч
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346254*:
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
:џџџџџџџџџ@к
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346268*J
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
:џџџџџџџџџЩ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346288*<
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
:џџџџџџџџџЩ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346302*<
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
:џџџџџџџџџ@Ч
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346316*:
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
Щ
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_88345283

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
К
H
,__inference_flatten_4_layer_call_fn_88346596

inputs
identityИ
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_88345283a
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
Ѕ
|
%__inference_internal_grad_fn_88347549
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

i
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88346510

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
­
љ
F__inference_dense_12_layer_call_and_return_conditional_losses_88345222

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
:џџџџџџџџџ@Ќ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345214*:
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
у

%__inference_internal_grad_fn_88347333
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
е

%__inference_internal_grad_fn_88347243
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
Щ	
ї
F__inference_dense_17_layer_call_and_return_conditional_losses_88346726

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


%__inference_internal_grad_fn_88346919
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
Ѕ
|
%__inference_internal_grad_fn_88346991
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
д

%__inference_internal_grad_fn_88347207
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

|
%__inference_internal_grad_fn_88347567
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


%__inference_internal_grad_fn_88346937
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
Я

G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149

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
:џџџџџџџџџ		О
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345141*L
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
Щ
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_88346602

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
Ю

+__inference_dense_16_layer_call_fn_88346689

inputs
unknown:	@
	unknown_0:@
identityЂStatefulPartitionedCallр
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_88345368o
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
­
|
%__inference_internal_grad_fn_88347657
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

 
%__inference_internal_grad_fn_88346829
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


%__inference_internal_grad_fn_88346883
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

|
%__inference_internal_grad_fn_88347081
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
О

G__inference_conv2d_14_layer_call_and_return_conditional_losses_88346564

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
:џџџџџџџџџ@М
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346556*J
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
у

%__inference_internal_grad_fn_88347369
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
­
љ
F__inference_dense_12_layer_call_and_return_conditional_losses_88346537

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
:џџџџџџџџџ@Ќ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346529*:
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
љ
Ѓ
,__inference_conv2d_10_layer_call_fn_88346391

inputs"
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallъ
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125x
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
ф

%__inference_internal_grad_fn_88347297
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
ДJ
М

E__inference_model_2_layer_call_and_return_conditional_losses_88345847
input_5
input_6-
conv2d_10_88345786:!
conv2d_10_88345788:	.
conv2d_11_88345791:!
conv2d_11_88345793:	.
conv2d_12_88345797:!
conv2d_12_88345799:	-
conv2d_13_88345802:@ 
conv2d_13_88345804:@#
dense_12_88345807:@
dense_12_88345809:@#
dense_13_88345813:@ 
dense_13_88345815: ,
conv2d_14_88345818:@@ 
conv2d_14_88345820:@%
dense_14_88345826:
  
dense_14_88345828:	%
dense_15_88345831:
 
dense_15_88345833:	$
dense_16_88345836:	@
dense_16_88345838:@#
dense_17_88345841:@
dense_17_88345843:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_88345786conv2d_10_88345788*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125Ќ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_88345791conv2d_11_88345793*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149њ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88345083Њ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_88345797conv2d_12_88345799*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174Ћ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_88345802conv2d_13_88345804*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198ќ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_88345807dense_12_88345809*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_88345222љ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88345095
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_88345813dense_13_88345815*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_88345247Љ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_88345818conv2d_14_88345820*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271ц
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_88345283ф
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_88345291
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
GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88345300
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_88345826dense_14_88345828*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_88345320
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_88345831dense_15_88345833*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_88345344
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_88345836dense_16_88345838*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_88345368
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_88345841dense_17_88345843*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_88345384x
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
у

%__inference_internal_grad_fn_88347351
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
у

%__inference_internal_grad_fn_88347171
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
­
|
%__inference_internal_grad_fn_88347117
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
ВJ
М

E__inference_model_2_layer_call_and_return_conditional_losses_88345391

inputs
inputs_1-
conv2d_10_88345126:!
conv2d_10_88345128:	.
conv2d_11_88345150:!
conv2d_11_88345152:	.
conv2d_12_88345175:!
conv2d_12_88345177:	-
conv2d_13_88345199:@ 
conv2d_13_88345201:@#
dense_12_88345223:@
dense_12_88345225:@#
dense_13_88345248:@ 
dense_13_88345250: ,
conv2d_14_88345272:@@ 
conv2d_14_88345274:@%
dense_14_88345321:
  
dense_14_88345323:	%
dense_15_88345345:
 
dense_15_88345347:	$
dense_16_88345369:	@
dense_16_88345371:@#
dense_17_88345385:@
dense_17_88345387:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_88345126conv2d_10_88345128*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125Ќ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_88345150conv2d_11_88345152*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149њ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88345083Њ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_88345175conv2d_12_88345177*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174Ћ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_88345199conv2d_13_88345201*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198§
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_88345223dense_12_88345225*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_88345222љ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88345095
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_88345248dense_13_88345250*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_88345247Љ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_88345272conv2d_14_88345274*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271ц
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_88345283ф
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_88345291
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
GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88345300
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_88345321dense_14_88345323*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_88345320
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_88345345dense_15_88345347*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_88345344
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_88345369dense_16_88345371*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_88345368
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_88345385dense_17_88345387*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_88345384x
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
­
|
%__inference_internal_grad_fn_88347639
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

i
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88345083

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
С
N
2__inference_max_pooling2d_4_layer_call_fn_88346441

inputs
identityр
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88345083
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

 
%__inference_internal_grad_fn_88346901
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
ѕ
Ё
,__inference_conv2d_14_layer_call_fn_88346546

inputs!
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallщ
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271w
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
ј
Ђ
,__inference_conv2d_13_layer_call_fn_88346482

inputs"
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallщ
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198w
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
ѕЕ
Ю
#__inference__wrapped_model_88345074
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
:џџџџџџџџџ		є
model_2/conv2d_10/IdentityN	IdentityNmodel_2/conv2d_10/mul_1:z:0"model_2/conv2d_10/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88344926*L
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
:џџџџџџџџџ		є
model_2/conv2d_11/IdentityN	IdentityNmodel_2/conv2d_11/mul_1:z:0"model_2/conv2d_11/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88344940*L
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
:џџџџџџџџџє
model_2/conv2d_12/IdentityN	IdentityNmodel_2/conv2d_12/mul_1:z:0"model_2/conv2d_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88344955*L
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
:џџџџџџџџџ@ђ
model_2/conv2d_13/IdentityN	IdentityNmodel_2/conv2d_13/mul_1:z:0"model_2/conv2d_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88344969*J
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
:џџџџџџџџџ@п
model_2/dense_12/IdentityN	IdentityNmodel_2/dense_12/mul_1:z:0!model_2/dense_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88344983*:
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
:џџџџџџџџџ п
model_2/dense_13/IdentityN	IdentityNmodel_2/dense_13/mul_1:z:0!model_2/dense_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88344998*:
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
:џџџџџџџџџ@ђ
model_2/conv2d_14/IdentityN	IdentityNmodel_2/conv2d_14/mul_1:z:0"model_2/conv2d_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345012*J
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
:џџџџџџџџџс
model_2/dense_14/IdentityN	IdentityNmodel_2/dense_14/mul_1:z:0!model_2/dense_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345032*<
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
:џџџџџџџџџс
model_2/dense_15/IdentityN	IdentityNmodel_2/dense_15/mul_1:z:0!model_2/dense_15/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345046*<
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
:џџџџџџџџџ@п
model_2/dense_16/IdentityN	IdentityNmodel_2/dense_16/mul_1:z:0!model_2/dense_16/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345060*:
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
д

%__inference_internal_grad_fn_88347387
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
в

E__inference_model_2_layer_call_and_return_conditional_losses_88346171
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
:џџџџџџџџџ		м
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346023*L
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
:џџџџџџџџџ		м
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346037*L
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
:џџџџџџџџџм
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346052*L
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
:џџџџџџџџџ@к
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346066*J
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
:џџџџџџџџџ@Ч
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346080*:
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
:џџџџџџџџџ Ч
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346095*:
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
:џџџџџџџџџ@к
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346109*J
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
:џџџџџџџџџЩ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346129*<
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
:џџџџџџџџџЩ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346143*<
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
:џџџџџџџџџ@Ч
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346157*:
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

|
%__inference_internal_grad_fn_88347063
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
Ш

&__inference_signature_wrapper_88346382
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
identityЂStatefulPartitionedCallз
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
GPU2*0,1J 8 *,
f'R%
#__inference__wrapped_model_88345074o
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
Й
\
0__inference_concatenate_2_layer_call_fn_88346619
inputs_0
inputs_1
identityЩ
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
GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88345300a
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
­
љ
F__inference_dense_13_layer_call_and_return_conditional_losses_88346591

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
:џџџџџџџџџ Ќ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346583*:
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
Т

G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198

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
:џџџџџџџџџ@М
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345190*J
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
О
ќ
F__inference_dense_14_layer_call_and_return_conditional_losses_88345320

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
:џџџџџџџџџЎ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345312*<
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
е

%__inference_internal_grad_fn_88347423
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

 
%__inference_internal_grad_fn_88346793
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
Б
њ
F__inference_dense_16_layer_call_and_return_conditional_losses_88345368

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
:џџџџџџџџџ@Ќ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345360*:
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
ДJ
М

E__inference_model_2_layer_call_and_return_conditional_losses_88345912
input_5
input_6-
conv2d_10_88345851:!
conv2d_10_88345853:	.
conv2d_11_88345856:!
conv2d_11_88345858:	.
conv2d_12_88345862:!
conv2d_12_88345864:	-
conv2d_13_88345867:@ 
conv2d_13_88345869:@#
dense_12_88345872:@
dense_12_88345874:@#
dense_13_88345878:@ 
dense_13_88345880: ,
conv2d_14_88345883:@@ 
conv2d_14_88345885:@%
dense_14_88345891:
  
dense_14_88345893:	%
dense_15_88345896:
 
dense_15_88345898:	$
dense_16_88345901:	@
dense_16_88345903:@#
dense_17_88345906:@
dense_17_88345908:
identityЂ!conv2d_10/StatefulPartitionedCallЂ!conv2d_11/StatefulPartitionedCallЂ!conv2d_12/StatefulPartitionedCallЂ!conv2d_13/StatefulPartitionedCallЂ!conv2d_14/StatefulPartitionedCallЂ dense_12/StatefulPartitionedCallЂ dense_13/StatefulPartitionedCallЂ dense_14/StatefulPartitionedCallЂ dense_15/StatefulPartitionedCallЂ dense_16/StatefulPartitionedCallЂ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_88345851conv2d_10_88345853*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125Ќ
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_88345856conv2d_11_88345858*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149њ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88345083Њ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_88345862conv2d_12_88345864*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174Ћ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_88345867conv2d_13_88345869*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198ќ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_88345872dense_12_88345874*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_88345222љ
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
GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88345095
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_88345878dense_13_88345880*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_88345247Љ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_88345883conv2d_14_88345885*
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271ц
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_88345283ф
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
GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_88345291
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
GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88345300
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_88345891dense_14_88345893*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_88345320
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_88345896dense_15_88345898*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_88345344
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_88345901dense_16_88345903*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_88345368
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_88345906dense_17_88345908*
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_88345384x
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
Я

G__inference_conv2d_11_layer_call_and_return_conditional_losses_88346436

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
:џџџџџџџџџ		О
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346428*L
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


%__inference_internal_grad_fn_88346955
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

|
%__inference_internal_grad_fn_88347603
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

|
%__inference_internal_grad_fn_88347135
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
є

*__inference_model_2_layer_call_fn_88346012
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
identityЂStatefulPartitionedCallћ
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
GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_88345685o
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
ю

*__inference_model_2_layer_call_fn_88345438
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
identityЂStatefulPartitionedCallљ
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
GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_88345391o
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
ќ
Є
,__inference_conv2d_12_layer_call_fn_88346455

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallъ
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174x
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
ф

%__inference_internal_grad_fn_88347279
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
у

%__inference_internal_grad_fn_88347153
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


%__inference_internal_grad_fn_88346865
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
ю

*__inference_model_2_layer_call_fn_88345782
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
identityЂStatefulPartitionedCallљ
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
GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_88345685o
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
е

%__inference_internal_grad_fn_88347225
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
ф

%__inference_internal_grad_fn_88347477
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
О
ќ
F__inference_dense_15_layer_call_and_return_conditional_losses_88345344

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
:џџџџџџџџџЎ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88345336*<
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
Ѕ
|
%__inference_internal_grad_fn_88347531
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
ќ
Є
,__inference_conv2d_11_layer_call_fn_88346418

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallъ
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
GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149x
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
Ы

+__inference_dense_13_layer_call_fn_88346573

inputs
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCallр
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
GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_88345247o
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
Т

G__inference_conv2d_13_layer_call_and_return_conditional_losses_88346500

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
:џџџџџџџџџ@М
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-88346492*J
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
З
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_88346613

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
ЁX
б
$__inference__traced_restore_88347792
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
_user_specified_namefile_prefix@
%__inference_internal_grad_fn_88346793CustomGradient-88344926@
%__inference_internal_grad_fn_88346811CustomGradient-88344940@
%__inference_internal_grad_fn_88346829CustomGradient-88344955@
%__inference_internal_grad_fn_88346847CustomGradient-88344969@
%__inference_internal_grad_fn_88346865CustomGradient-88344983@
%__inference_internal_grad_fn_88346883CustomGradient-88344998@
%__inference_internal_grad_fn_88346901CustomGradient-88345012@
%__inference_internal_grad_fn_88346919CustomGradient-88345032@
%__inference_internal_grad_fn_88346937CustomGradient-88345046@
%__inference_internal_grad_fn_88346955CustomGradient-88345060@
%__inference_internal_grad_fn_88346973CustomGradient-88345117@
%__inference_internal_grad_fn_88346991CustomGradient-88345141@
%__inference_internal_grad_fn_88347009CustomGradient-88345166@
%__inference_internal_grad_fn_88347027CustomGradient-88345190@
%__inference_internal_grad_fn_88347045CustomGradient-88345214@
%__inference_internal_grad_fn_88347063CustomGradient-88345239@
%__inference_internal_grad_fn_88347081CustomGradient-88345263@
%__inference_internal_grad_fn_88347099CustomGradient-88345312@
%__inference_internal_grad_fn_88347117CustomGradient-88345336@
%__inference_internal_grad_fn_88347135CustomGradient-88345360@
%__inference_internal_grad_fn_88347153CustomGradient-88346023@
%__inference_internal_grad_fn_88347171CustomGradient-88346037@
%__inference_internal_grad_fn_88347189CustomGradient-88346052@
%__inference_internal_grad_fn_88347207CustomGradient-88346066@
%__inference_internal_grad_fn_88347225CustomGradient-88346080@
%__inference_internal_grad_fn_88347243CustomGradient-88346095@
%__inference_internal_grad_fn_88347261CustomGradient-88346109@
%__inference_internal_grad_fn_88347279CustomGradient-88346129@
%__inference_internal_grad_fn_88347297CustomGradient-88346143@
%__inference_internal_grad_fn_88347315CustomGradient-88346157@
%__inference_internal_grad_fn_88347333CustomGradient-88346182@
%__inference_internal_grad_fn_88347351CustomGradient-88346196@
%__inference_internal_grad_fn_88347369CustomGradient-88346211@
%__inference_internal_grad_fn_88347387CustomGradient-88346225@
%__inference_internal_grad_fn_88347405CustomGradient-88346239@
%__inference_internal_grad_fn_88347423CustomGradient-88346254@
%__inference_internal_grad_fn_88347441CustomGradient-88346268@
%__inference_internal_grad_fn_88347459CustomGradient-88346288@
%__inference_internal_grad_fn_88347477CustomGradient-88346302@
%__inference_internal_grad_fn_88347495CustomGradient-88346316@
%__inference_internal_grad_fn_88347513CustomGradient-88346401@
%__inference_internal_grad_fn_88347531CustomGradient-88346428@
%__inference_internal_grad_fn_88347549CustomGradient-88346465@
%__inference_internal_grad_fn_88347567CustomGradient-88346492@
%__inference_internal_grad_fn_88347585CustomGradient-88346529@
%__inference_internal_grad_fn_88347603CustomGradient-88346556@
%__inference_internal_grad_fn_88347621CustomGradient-88346583@
%__inference_internal_grad_fn_88347639CustomGradient-88346645@
%__inference_internal_grad_fn_88347657CustomGradient-88346672@
%__inference_internal_grad_fn_88347675CustomGradient-88346699"лL
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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:з
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
і2ѓ
*__inference_model_2_layer_call_fn_88345438
*__inference_model_2_layer_call_fn_88345962
*__inference_model_2_layer_call_fn_88346012
*__inference_model_2_layer_call_fn_88345782Р
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
т2п
E__inference_model_2_layer_call_and_return_conditional_losses_88346171
E__inference_model_2_layer_call_and_return_conditional_losses_88346330
E__inference_model_2_layer_call_and_return_conditional_losses_88345847
E__inference_model_2_layer_call_and_return_conditional_losses_88345912Р
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
зBд
#__inference__wrapped_model_88345074input_5input_6"
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
ж2г
,__inference_conv2d_10_layer_call_fn_88346391Ђ
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
G__inference_conv2d_10_layer_call_and_return_conditional_losses_88346409Ђ
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
ж2г
,__inference_conv2d_11_layer_call_fn_88346418Ђ
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
G__inference_conv2d_11_layer_call_and_return_conditional_losses_88346436Ђ
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
м2й
2__inference_max_pooling2d_4_layer_call_fn_88346441Ђ
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
ї2є
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88346446Ђ
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
ж2г
,__inference_conv2d_12_layer_call_fn_88346455Ђ
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
G__inference_conv2d_12_layer_call_and_return_conditional_losses_88346473Ђ
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
ж2г
,__inference_conv2d_13_layer_call_fn_88346482Ђ
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
G__inference_conv2d_13_layer_call_and_return_conditional_losses_88346500Ђ
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
м2й
2__inference_max_pooling2d_5_layer_call_fn_88346505Ђ
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
ї2є
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88346510Ђ
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
е2в
+__inference_dense_12_layer_call_fn_88346519Ђ
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
№2э
F__inference_dense_12_layer_call_and_return_conditional_losses_88346537Ђ
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
ж2г
,__inference_conv2d_14_layer_call_fn_88346546Ђ
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
G__inference_conv2d_14_layer_call_and_return_conditional_losses_88346564Ђ
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
е2в
+__inference_dense_13_layer_call_fn_88346573Ђ
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
№2э
F__inference_dense_13_layer_call_and_return_conditional_losses_88346591Ђ
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
ж2г
,__inference_flatten_4_layer_call_fn_88346596Ђ
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
G__inference_flatten_4_layer_call_and_return_conditional_losses_88346602Ђ
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
ж2г
,__inference_flatten_5_layer_call_fn_88346607Ђ
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
G__inference_flatten_5_layer_call_and_return_conditional_losses_88346613Ђ
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
к2з
0__inference_concatenate_2_layer_call_fn_88346619Ђ
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
ѕ2ђ
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88346626Ђ
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
е2в
+__inference_dense_14_layer_call_fn_88346635Ђ
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
№2э
F__inference_dense_14_layer_call_and_return_conditional_losses_88346653Ђ
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
е2в
+__inference_dense_15_layer_call_fn_88346662Ђ
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
№2э
F__inference_dense_15_layer_call_and_return_conditional_losses_88346680Ђ
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
е2в
+__inference_dense_16_layer_call_fn_88346689Ђ
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
№2э
F__inference_dense_16_layer_call_and_return_conditional_losses_88346707Ђ
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
е2в
+__inference_dense_17_layer_call_fn_88346716Ђ
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
№2э
F__inference_dense_17_layer_call_and_return_conditional_losses_88346726Ђ
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
дBб
&__inference_signature_wrapper_88346382input_5input_6"
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
Ab?
model_2/conv2d_10/beta:0#__inference__wrapped_model_88345074
DbB
model_2/conv2d_10/BiasAdd:0#__inference__wrapped_model_88345074
Ab?
model_2/conv2d_11/beta:0#__inference__wrapped_model_88345074
DbB
model_2/conv2d_11/BiasAdd:0#__inference__wrapped_model_88345074
Ab?
model_2/conv2d_12/beta:0#__inference__wrapped_model_88345074
DbB
model_2/conv2d_12/BiasAdd:0#__inference__wrapped_model_88345074
Ab?
model_2/conv2d_13/beta:0#__inference__wrapped_model_88345074
DbB
model_2/conv2d_13/BiasAdd:0#__inference__wrapped_model_88345074
@b>
model_2/dense_12/beta:0#__inference__wrapped_model_88345074
CbA
model_2/dense_12/BiasAdd:0#__inference__wrapped_model_88345074
@b>
model_2/dense_13/beta:0#__inference__wrapped_model_88345074
CbA
model_2/dense_13/BiasAdd:0#__inference__wrapped_model_88345074
Ab?
model_2/conv2d_14/beta:0#__inference__wrapped_model_88345074
DbB
model_2/conv2d_14/BiasAdd:0#__inference__wrapped_model_88345074
@b>
model_2/dense_14/beta:0#__inference__wrapped_model_88345074
CbA
model_2/dense_14/BiasAdd:0#__inference__wrapped_model_88345074
@b>
model_2/dense_15/beta:0#__inference__wrapped_model_88345074
CbA
model_2/dense_15/BiasAdd:0#__inference__wrapped_model_88345074
@b>
model_2/dense_16/beta:0#__inference__wrapped_model_88345074
CbA
model_2/dense_16/BiasAdd:0#__inference__wrapped_model_88345074
SbQ
beta:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125
VbT
	BiasAdd:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_88345125
SbQ
beta:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149
VbT
	BiasAdd:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_88345149
SbQ
beta:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174
VbT
	BiasAdd:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_88345174
SbQ
beta:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198
VbT
	BiasAdd:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_88345198
RbP
beta:0F__inference_dense_12_layer_call_and_return_conditional_losses_88345222
UbS
	BiasAdd:0F__inference_dense_12_layer_call_and_return_conditional_losses_88345222
RbP
beta:0F__inference_dense_13_layer_call_and_return_conditional_losses_88345247
UbS
	BiasAdd:0F__inference_dense_13_layer_call_and_return_conditional_losses_88345247
SbQ
beta:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271
VbT
	BiasAdd:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_88345271
RbP
beta:0F__inference_dense_14_layer_call_and_return_conditional_losses_88345320
UbS
	BiasAdd:0F__inference_dense_14_layer_call_and_return_conditional_losses_88345320
RbP
beta:0F__inference_dense_15_layer_call_and_return_conditional_losses_88345344
UbS
	BiasAdd:0F__inference_dense_15_layer_call_and_return_conditional_losses_88345344
RbP
beta:0F__inference_dense_16_layer_call_and_return_conditional_losses_88345368
UbS
	BiasAdd:0F__inference_dense_16_layer_call_and_return_conditional_losses_88345368
[bY
conv2d_10/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
^b\
conv2d_10/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
[bY
conv2d_11/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
^b\
conv2d_11/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
[bY
conv2d_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
^b\
conv2d_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
[bY
conv2d_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
^b\
conv2d_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
ZbX
dense_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
]b[
dense_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
ZbX
dense_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
]b[
dense_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
[bY
conv2d_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
^b\
conv2d_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
ZbX
dense_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
]b[
dense_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
ZbX
dense_15/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
]b[
dense_15/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
ZbX
dense_16/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
]b[
dense_16/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346171
[bY
conv2d_10/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
^b\
conv2d_10/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
[bY
conv2d_11/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
^b\
conv2d_11/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
[bY
conv2d_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
^b\
conv2d_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
[bY
conv2d_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
^b\
conv2d_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
ZbX
dense_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
]b[
dense_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
ZbX
dense_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
]b[
dense_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
[bY
conv2d_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
^b\
conv2d_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
ZbX
dense_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
]b[
dense_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
ZbX
dense_15/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
]b[
dense_15/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
ZbX
dense_16/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
]b[
dense_16/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_88346330
SbQ
beta:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_88346409
VbT
	BiasAdd:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_88346409
SbQ
beta:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_88346436
VbT
	BiasAdd:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_88346436
SbQ
beta:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_88346473
VbT
	BiasAdd:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_88346473
SbQ
beta:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_88346500
VbT
	BiasAdd:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_88346500
RbP
beta:0F__inference_dense_12_layer_call_and_return_conditional_losses_88346537
UbS
	BiasAdd:0F__inference_dense_12_layer_call_and_return_conditional_losses_88346537
SbQ
beta:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_88346564
VbT
	BiasAdd:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_88346564
RbP
beta:0F__inference_dense_13_layer_call_and_return_conditional_losses_88346591
UbS
	BiasAdd:0F__inference_dense_13_layer_call_and_return_conditional_losses_88346591
RbP
beta:0F__inference_dense_14_layer_call_and_return_conditional_losses_88346653
UbS
	BiasAdd:0F__inference_dense_14_layer_call_and_return_conditional_losses_88346653
RbP
beta:0F__inference_dense_15_layer_call_and_return_conditional_losses_88346680
UbS
	BiasAdd:0F__inference_dense_15_layer_call_and_return_conditional_losses_88346680
RbP
beta:0F__inference_dense_16_layer_call_and_return_conditional_losses_88346707
UbS
	BiasAdd:0F__inference_dense_16_layer_call_and_return_conditional_losses_88346707п
#__inference__wrapped_model_88345074З&'67?@PQbcYZ`Ђ]
VЂS
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
Њ "3Њ0
.
dense_17"
dense_17џџџџџџџџџе
K__inference_concatenate_2_layer_call_and_return_conditional_losses_88346626[ЂX
QЂN
LI
# 
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџ 
 Ќ
0__inference_concatenate_2_layer_call_fn_88346619x[ЂX
QЂN
LI
# 
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ 
Њ "џџџџџџџџџ И
G__inference_conv2d_10_layer_call_and_return_conditional_losses_88346409m7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		
Њ ".Ђ+
$!
0џџџџџџџџџ		
 
,__inference_conv2d_10_layer_call_fn_88346391`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		
Њ "!џџџџџџџџџ		Й
G__inference_conv2d_11_layer_call_and_return_conditional_losses_88346436n&'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ		
Њ ".Ђ+
$!
0џџџџџџџџџ		
 
,__inference_conv2d_11_layer_call_fn_88346418a&'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ		
Њ "!џџџџџџџџџ		Й
G__inference_conv2d_12_layer_call_and_return_conditional_losses_88346473n678Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
,__inference_conv2d_12_layer_call_fn_88346455a678Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџИ
G__inference_conv2d_13_layer_call_and_return_conditional_losses_88346500m?@8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
,__inference_conv2d_13_layer_call_fn_88346482`?@8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ " џџџџџџџџџ@З
G__inference_conv2d_14_layer_call_and_return_conditional_losses_88346564lYZ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
,__inference_conv2d_14_layer_call_fn_88346546_YZ7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ " џџџџџџџџџ@І
F__inference_dense_12_layer_call_and_return_conditional_losses_88346537\PQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 ~
+__inference_dense_12_layer_call_fn_88346519OPQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@І
F__inference_dense_13_layer_call_and_return_conditional_losses_88346591\bc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ 
 ~
+__inference_dense_13_layer_call_fn_88346573Obc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ Њ
F__inference_dense_14_layer_call_and_return_conditional_losses_88346653`0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџ
 
+__inference_dense_14_layer_call_fn_88346635S0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЊ
F__inference_dense_15_layer_call_and_return_conditional_losses_88346680`0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
+__inference_dense_15_layer_call_fn_88346662S0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЉ
F__inference_dense_16_layer_call_and_return_conditional_losses_88346707_0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 
+__inference_dense_16_layer_call_fn_88346689R0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ј
F__inference_dense_17_layer_call_and_return_conditional_losses_88346726^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_dense_17_layer_call_fn_88346716Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЌ
G__inference_flatten_4_layer_call_and_return_conditional_losses_88346602a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 
,__inference_flatten_4_layer_call_fn_88346596T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЃ
G__inference_flatten_5_layer_call_and_return_conditional_losses_88346613X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 {
,__inference_flatten_5_layer_call_fn_88346607K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ и
%__inference_internal_grad_fn_88346793ЎњћwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88346811Ўќ§wЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88346829ЎўџwЂt
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
1џџџџџџџџџе
%__inference_internal_grad_fn_88346847ЋuЂr
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88346865eЂb
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88346883eЂb
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
1џџџџџџџџџ е
%__inference_internal_grad_fn_88346901ЋuЂr
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
1џџџџџџџџџ@Р
%__inference_internal_grad_fn_88346919gЂd
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
1џџџџџџџџџР
%__inference_internal_grad_fn_88346937gЂd
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
1џџџџџџџџџН
%__inference_internal_grad_fn_88346955eЂb
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
1џџџџџџџџџ@и
%__inference_internal_grad_fn_88346973ЎwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88346991ЎwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88347009ЎwЂt
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
1џџџџџџџџџе
%__inference_internal_grad_fn_88347027ЋuЂr
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347045eЂb
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347063eЂb
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
1џџџџџџџџџ е
%__inference_internal_grad_fn_88347081ЋuЂr
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
1џџџџџџџџџ@Р
%__inference_internal_grad_fn_88347099gЂd
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
1џџџџџџџџџР
%__inference_internal_grad_fn_88347117gЂd
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
1џџџџџџџџџН
%__inference_internal_grad_fn_88347135 ЁeЂb
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
1џџџџџџџџџ@и
%__inference_internal_grad_fn_88347153ЎЂЃwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88347171ЎЄЅwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88347189ЎІЇwЂt
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
1џџџџџџџџџе
%__inference_internal_grad_fn_88347207ЋЈЉuЂr
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347225ЊЋeЂb
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347243Ќ­eЂb
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
1џџџџџџџџџ е
%__inference_internal_grad_fn_88347261ЋЎЏuЂr
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
1џџџџџџџџџ@Р
%__inference_internal_grad_fn_88347279АБgЂd
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
1џџџџџџџџџР
%__inference_internal_grad_fn_88347297ВГgЂd
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
1џџџџџџџџџН
%__inference_internal_grad_fn_88347315ДЕeЂb
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
1џџџџџџџџџ@и
%__inference_internal_grad_fn_88347333ЎЖЗwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88347351ЎИЙwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88347369ЎКЛwЂt
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
1џџџџџџџџџе
%__inference_internal_grad_fn_88347387ЋМНuЂr
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347405ОПeЂb
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347423РСeЂb
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
1џџџџџџџџџ е
%__inference_internal_grad_fn_88347441ЋТУuЂr
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
1џџџџџџџџџ@Р
%__inference_internal_grad_fn_88347459ФХgЂd
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
1џџџџџџџџџР
%__inference_internal_grad_fn_88347477ЦЧgЂd
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
1џџџџџџџџџН
%__inference_internal_grad_fn_88347495ШЩeЂb
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
1џџџџџџџџџ@и
%__inference_internal_grad_fn_88347513ЎЪЫwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88347531ЎЬЭwЂt
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
1џџџџџџџџџ		и
%__inference_internal_grad_fn_88347549ЎЮЯwЂt
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
1џџџџџџџџџе
%__inference_internal_grad_fn_88347567ЋабuЂr
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347585вгeЂb
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
1џџџџџџџџџ@е
%__inference_internal_grad_fn_88347603ЋдеuЂr
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
1џџџџџџџџџ@Н
%__inference_internal_grad_fn_88347621жзeЂb
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
1џџџџџџџџџ Р
%__inference_internal_grad_fn_88347639ийgЂd
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
1џџџџџџџџџР
%__inference_internal_grad_fn_88347657клgЂd
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
1џџџџџџџџџН
%__inference_internal_grad_fn_88347675мнeЂb
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
1џџџџџџџџџ@№
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_88346446RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
2__inference_max_pooling2d_4_layer_call_fn_88346441RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ№
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_88346510RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ш
2__inference_max_pooling2d_5_layer_call_fn_88346505RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџћ
E__inference_model_2_layer_call_and_return_conditional_losses_88345847Б&'67?@PQbcYZhЂe
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
 ћ
E__inference_model_2_layer_call_and_return_conditional_losses_88345912Б&'67?@PQbcYZhЂe
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
 §
E__inference_model_2_layer_call_and_return_conditional_losses_88346171Г&'67?@PQbcYZjЂg
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
 §
E__inference_model_2_layer_call_and_return_conditional_losses_88346330Г&'67?@PQbcYZjЂg
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
 г
*__inference_model_2_layer_call_fn_88345438Є&'67?@PQbcYZhЂe
^Ђ[
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
p 

 
Њ "џџџџџџџџџг
*__inference_model_2_layer_call_fn_88345782Є&'67?@PQbcYZhЂe
^Ђ[
QN
)&
input_5џџџџџџџџџ		
!
input_6џџџџџџџџџ
p

 
Њ "џџџџџџџџџе
*__inference_model_2_layer_call_fn_88345962І&'67?@PQbcYZjЂg
`Ђ]
SP
*'
inputs/0џџџџџџџџџ		
"
inputs/1џџџџџџџџџ
p 

 
Њ "џџџџџџџџџе
*__inference_model_2_layer_call_fn_88346012І&'67?@PQbcYZjЂg
`Ђ]
SP
*'
inputs/0џџџџџџџџџ		
"
inputs/1џџџџџџџџџ
p

 
Њ "џџџџџџџџџѓ
&__inference_signature_wrapper_88346382Ш&'67?@PQbcYZqЂn
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