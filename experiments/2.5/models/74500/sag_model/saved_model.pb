Å¦
¢ó
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
Á
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
executor_typestring ¨
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ù
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
Êc
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*c
valueûbBøb Bñb
ó
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
Ë

kernel
bias
#_self_saveable_object_factories
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
Ë

&kernel
'bias
#(_self_saveable_object_factories
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
³
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
Ë

6kernel
7bias
#8_self_saveable_object_factories
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
Ë

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
³
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
Ë

Pkernel
Qbias
#R_self_saveable_object_factories
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*
Ë

Ykernel
Zbias
#[_self_saveable_object_factories
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*
Ë

bkernel
cbias
#d_self_saveable_object_factories
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses*
³
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
³
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses* 
³
#y_self_saveable_object_factories
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
Ô
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ô
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ô
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ô
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢__call__
+£&call_and_return_all_conditional_losses*

¤serving_default* 
* 
²
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
²
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
µ
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
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
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
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
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
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
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
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
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
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
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
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
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
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
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
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
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
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
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
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
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
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
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
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
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
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
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
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
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
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
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
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
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
 regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses*
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
:ÿÿÿÿÿÿÿÿÿ		*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ		
z
serving_default_input_6Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
é
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_6conv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasconv2d_14/kernelconv2d_14/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 */
f*R(
&__inference_signature_wrapper_96518578
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ú
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
!__inference__traced_save_96519912
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
$__inference__traced_restore_96519988ÓÕ


%__inference_internal_grad_fn_96519133
result_grads_0
result_grads_1
mul_model_2_dense_15_beta 
mul_model_2_dense_15_biasadd
identity
mulMulmul_model_2_dense_15_betamul_model_2_dense_15_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
mul_1Mulmul_model_2_dense_15_betamul_model_2_dense_15_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ô

%__inference_internal_grad_fn_96519457
result_grads_0
result_grads_1
mul_conv2d_14_beta
mul_conv2d_14_biasadd
identity
mulMulmul_conv2d_14_betamul_conv2d_14_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
mul_1Mulmul_conv2d_14_betamul_conv2d_14_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
º
H
,__inference_flatten_4_layer_call_fn_96518792

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_96517479a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô

*__inference_model_2_layer_call_fn_96518158
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
identity¢StatefulPartitionedCallû
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
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_96517587o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
­
ù
F__inference_dense_13_layer_call_and_return_conditional_losses_96518787

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518779*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë

+__inference_dense_17_layer_call_fn_96518912

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_96517580o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


%__inference_internal_grad_fn_96519151
result_grads_0
result_grads_1
mul_model_2_dense_16_beta 
mul_model_2_dense_16_biasadd
identity
mulMulmul_model_2_dense_16_betamul_model_2_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
mul_1Mulmul_model_2_dense_16_betamul_model_2_dense_16_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
õ
¡
,__inference_conv2d_14_layer_call_fn_96518742

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

 
%__inference_internal_grad_fn_96519007
result_grads_0
result_grads_1
mul_model_2_conv2d_11_beta!
mul_model_2_conv2d_11_biasadd
identity
mulMulmul_model_2_conv2d_11_betamul_model_2_conv2d_11_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
mul_1Mulmul_model_2_conv2d_11_betamul_model_2_conv2d_11_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
Ï

G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		¾
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517337*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
­
ù
F__inference_dense_12_layer_call_and_return_conditional_losses_96518733

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518725*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

E__inference_model_2_layer_call_and_return_conditional_losses_96518367
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
identity¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp¢ conv2d_12/BiasAdd/ReadVariableOp¢conv2d_12/Conv2D/ReadVariableOp¢ conv2d_13/BiasAdd/ReadVariableOp¢conv2d_13/Conv2D/ReadVariableOp¢ conv2d_14/BiasAdd/ReadVariableOp¢conv2d_14/Conv2D/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0°
conv2d_10/Conv2DConv2Dinputs_0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		S
conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_10/mulMulconv2d_10/beta:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		j
conv2d_10/SigmoidSigmoidconv2d_10/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
conv2d_10/mul_1Mulconv2d_10/BiasAdd:output:0conv2d_10/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		n
conv2d_10/IdentityIdentityconv2d_10/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ü
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518219*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
conv2d_11/Conv2DConv2Dconv2d_10/IdentityN:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		S
conv2d_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_11/mulMulconv2d_11/beta:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		j
conv2d_11/SigmoidSigmoidconv2d_11/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
conv2d_11/mul_1Mulconv2d_11/BiasAdd:output:0conv2d_11/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		n
conv2d_11/IdentityIdentityconv2d_11/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ü
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518233*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		®
max_pooling2d_4/MaxPoolMaxPoolconv2d_11/IdentityN:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0È
conv2d_12/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿS
conv2d_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_12/mulMulconv2d_12/beta:output:0conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
conv2d_12/SigmoidSigmoidconv2d_12/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_12/mul_1Mulconv2d_12/BiasAdd:output:0conv2d_12/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
conv2d_12/IdentityIdentityconv2d_12/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518248*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ã
conv2d_13/Conv2DConv2Dconv2d_12/IdentityN:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@S
conv2d_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_13/mulMulconv2d_13/beta:output:0conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
conv2d_13/SigmoidSigmoidconv2d_13/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_13/mul_1Mulconv2d_13/BiasAdd:output:0conv2d_13/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
conv2d_13/IdentityIdentityconv2d_13/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518262*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0}
dense_12/MatMulMatMulinputs_1&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_12/mulMuldense_12/beta:output:0dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
dense_12/SigmoidSigmoiddense_12/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
dense_12/mul_1Muldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
dense_12/IdentityIdentitydense_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518276*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d_5/MaxPoolMaxPoolconv2d_13/IdentityN:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_13/mulMuldense_13/beta:output:0dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
dense_13/SigmoidSigmoiddense_13/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
dense_13/mul_1Muldense_13/BiasAdd:output:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_13/IdentityIdentitydense_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518291*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ 
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ç
conv2d_14/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@S
conv2d_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_14/mulMulconv2d_14/beta:output:0conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
conv2d_14/SigmoidSigmoidconv2d_14/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_14/mul_1Mulconv2d_14/BiasAdd:output:0conv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
conv2d_14/IdentityIdentityconv2d_14/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518305*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_4/ReshapeReshapeconv2d_14/IdentityN:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_5/ReshapeReshapedense_13/IdentityN:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¸
concatenate_2/concatConcatV2flatten_4/Reshape:output:0flatten_5/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_14/MatMulMatMulconcatenate_2/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_14/mulMuldense_14/beta:output:0dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_14/SigmoidSigmoiddense_14/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
dense_14/mul_1Muldense_14/BiasAdd:output:0dense_14/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/IdentityIdentitydense_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518325*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_15/MatMulMatMuldense_14/IdentityN:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_15/mulMuldense_15/beta:output:0dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_15/SigmoidSigmoiddense_15/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
dense_15/mul_1Muldense_15/BiasAdd:output:0dense_15/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_15/IdentityIdentitydense_15/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518339*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_16/MatMulMatMuldense_15/IdentityN:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_16/mulMuldense_16/beta:output:0dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
dense_16/SigmoidSigmoiddense_16/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
dense_16/mul_1Muldense_16/BiasAdd:output:0dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
dense_16/IdentityIdentitydense_16/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518353*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_17/MatMulMatMuldense_16/IdentityN:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2D
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
:ÿÿÿÿÿÿÿÿÿ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

|
%__inference_internal_grad_fn_96519799
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

i
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96518642

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

%__inference_internal_grad_fn_96519619
result_grads_0
result_grads_1
mul_dense_13_beta
mul_dense_13_biasadd
identityv
mulMulmul_dense_13_betamul_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
mul_1Mulmul_dense_13_betamul_dense_13_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :ÿÿÿÿÿÿÿÿÿ :W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
À
u
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96517496

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
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Õ

%__inference_internal_grad_fn_96519691
result_grads_0
result_grads_1
mul_dense_16_beta
mul_dense_16_biasadd
identityv
mulMulmul_dense_16_betamul_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
mul_1Mulmul_dense_16_betamul_dense_16_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
¾
ü
F__inference_dense_14_layer_call_and_return_conditional_losses_96517516

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517508*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

|
%__inference_internal_grad_fn_96519781
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@


%__inference_internal_grad_fn_96519079
result_grads_0
result_grads_1
mul_model_2_dense_13_beta 
mul_model_2_dense_13_biasadd
identity
mulMulmul_model_2_dense_13_betamul_model_2_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
mul_1Mulmul_model_2_dense_13_betamul_model_2_dense_13_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :ÿÿÿÿÿÿÿÿÿ :W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
¥
|
%__inference_internal_grad_fn_96519205
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
É
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_96517479

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­
|
%__inference_internal_grad_fn_96519853
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ä

%__inference_internal_grad_fn_96519655
result_grads_0
result_grads_1
mul_dense_14_beta
mul_dense_14_biasadd
identityw
mulMulmul_dense_14_betamul_dense_14_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
mul_1Mulmul_dense_14_betamul_dense_14_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

|
%__inference_internal_grad_fn_96519331
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
õµ
Î
#__inference__wrapped_model_96517270
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
identity¢(model_2/conv2d_10/BiasAdd/ReadVariableOp¢'model_2/conv2d_10/Conv2D/ReadVariableOp¢(model_2/conv2d_11/BiasAdd/ReadVariableOp¢'model_2/conv2d_11/Conv2D/ReadVariableOp¢(model_2/conv2d_12/BiasAdd/ReadVariableOp¢'model_2/conv2d_12/Conv2D/ReadVariableOp¢(model_2/conv2d_13/BiasAdd/ReadVariableOp¢'model_2/conv2d_13/Conv2D/ReadVariableOp¢(model_2/conv2d_14/BiasAdd/ReadVariableOp¢'model_2/conv2d_14/Conv2D/ReadVariableOp¢'model_2/dense_12/BiasAdd/ReadVariableOp¢&model_2/dense_12/MatMul/ReadVariableOp¢'model_2/dense_13/BiasAdd/ReadVariableOp¢&model_2/dense_13/MatMul/ReadVariableOp¢'model_2/dense_14/BiasAdd/ReadVariableOp¢&model_2/dense_14/MatMul/ReadVariableOp¢'model_2/dense_15/BiasAdd/ReadVariableOp¢&model_2/dense_15/MatMul/ReadVariableOp¢'model_2/dense_16/BiasAdd/ReadVariableOp¢&model_2/dense_16/MatMul/ReadVariableOp¢'model_2/dense_17/BiasAdd/ReadVariableOp¢&model_2/dense_17/MatMul/ReadVariableOp¡
'model_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0¿
model_2/conv2d_10/Conv2DConv2Dinput_5/model_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

(model_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
model_2/conv2d_10/BiasAddBiasAdd!model_2/conv2d_10/Conv2D:output:00model_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
model_2/conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_10/mulMulmodel_2/conv2d_10/beta:output:0"model_2/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		z
model_2/conv2d_10/SigmoidSigmoidmodel_2/conv2d_10/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
model_2/conv2d_10/mul_1Mul"model_2/conv2d_10/BiasAdd:output:0model_2/conv2d_10/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		~
model_2/conv2d_10/IdentityIdentitymodel_2/conv2d_10/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		ô
model_2/conv2d_10/IdentityN	IdentityNmodel_2/conv2d_10/mul_1:z:0"model_2/conv2d_10/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517122*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		¢
'model_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
model_2/conv2d_11/Conv2DConv2D$model_2/conv2d_10/IdentityN:output:0/model_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

(model_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
model_2/conv2d_11/BiasAddBiasAdd!model_2/conv2d_11/Conv2D:output:00model_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
model_2/conv2d_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_11/mulMulmodel_2/conv2d_11/beta:output:0"model_2/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		z
model_2/conv2d_11/SigmoidSigmoidmodel_2/conv2d_11/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
model_2/conv2d_11/mul_1Mul"model_2/conv2d_11/BiasAdd:output:0model_2/conv2d_11/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		~
model_2/conv2d_11/IdentityIdentitymodel_2/conv2d_11/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		ô
model_2/conv2d_11/IdentityN	IdentityNmodel_2/conv2d_11/mul_1:z:0"model_2/conv2d_11/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517136*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		¾
model_2/max_pooling2d_4/MaxPoolMaxPool$model_2/conv2d_11/IdentityN:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¢
'model_2/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0à
model_2/conv2d_12/Conv2DConv2D(model_2/max_pooling2d_4/MaxPool:output:0/model_2/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(model_2/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0´
model_2/conv2d_12/BiasAddBiasAdd!model_2/conv2d_12/Conv2D:output:00model_2/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
model_2/conv2d_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_12/mulMulmodel_2/conv2d_12/beta:output:0"model_2/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
model_2/conv2d_12/SigmoidSigmoidmodel_2/conv2d_12/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_2/conv2d_12/mul_1Mul"model_2/conv2d_12/BiasAdd:output:0model_2/conv2d_12/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
model_2/conv2d_12/IdentityIdentitymodel_2/conv2d_12/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
model_2/conv2d_12/IdentityN	IdentityNmodel_2/conv2d_12/mul_1:z:0"model_2/conv2d_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517151*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ¡
'model_2/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Û
model_2/conv2d_13/Conv2DConv2D$model_2/conv2d_12/IdentityN:output:0/model_2/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_2/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_2/conv2d_13/BiasAddBiasAdd!model_2/conv2d_13/Conv2D:output:00model_2/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
model_2/conv2d_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_13/mulMulmodel_2/conv2d_13/beta:output:0"model_2/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
model_2/conv2d_13/SigmoidSigmoidmodel_2/conv2d_13/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_2/conv2d_13/mul_1Mul"model_2/conv2d_13/BiasAdd:output:0model_2/conv2d_13/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
model_2/conv2d_13/IdentityIdentitymodel_2/conv2d_13/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ò
model_2/conv2d_13/IdentityN	IdentityNmodel_2/conv2d_13/mul_1:z:0"model_2/conv2d_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517165*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
&model_2/dense_12/MatMul/ReadVariableOpReadVariableOp/model_2_dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
model_2/dense_12/MatMulMatMulinput_6.model_2/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'model_2/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
model_2/dense_12/BiasAddBiasAdd!model_2/dense_12/MatMul:product:0/model_2/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
model_2/dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_12/mulMulmodel_2/dense_12/beta:output:0!model_2/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
model_2/dense_12/SigmoidSigmoidmodel_2/dense_12/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_2/dense_12/mul_1Mul!model_2/dense_12/BiasAdd:output:0model_2/dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
model_2/dense_12/IdentityIdentitymodel_2/dense_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
model_2/dense_12/IdentityN	IdentityNmodel_2/dense_12/mul_1:z:0!model_2/dense_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517179*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@½
model_2/max_pooling2d_5/MaxPoolMaxPool$model_2/conv2d_13/IdentityN:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

&model_2/dense_13/MatMul/ReadVariableOpReadVariableOp/model_2_dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0¨
model_2/dense_13/MatMulMatMul#model_2/dense_12/IdentityN:output:0.model_2/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'model_2/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
model_2/dense_13/BiasAddBiasAdd!model_2/dense_13/MatMul:product:0/model_2/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Z
model_2/dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_13/mulMulmodel_2/dense_13/beta:output:0!model_2/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ o
model_2/dense_13/SigmoidSigmoidmodel_2/dense_13/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model_2/dense_13/mul_1Mul!model_2/dense_13/BiasAdd:output:0model_2/dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ s
model_2/dense_13/IdentityIdentitymodel_2/dense_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ß
model_2/dense_13/IdentityN	IdentityNmodel_2/dense_13/mul_1:z:0!model_2/dense_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517194*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ  
'model_2/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ß
model_2/conv2d_14/Conv2DConv2D(model_2/max_pooling2d_5/MaxPool:output:0/model_2/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(model_2/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
model_2/conv2d_14/BiasAddBiasAdd!model_2/conv2d_14/Conv2D:output:00model_2/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
model_2/conv2d_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/conv2d_14/mulMulmodel_2/conv2d_14/beta:output:0"model_2/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@y
model_2/conv2d_14/SigmoidSigmoidmodel_2/conv2d_14/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_2/conv2d_14/mul_1Mul"model_2/conv2d_14/BiasAdd:output:0model_2/conv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@}
model_2/conv2d_14/IdentityIdentitymodel_2/conv2d_14/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ò
model_2/conv2d_14/IdentityN	IdentityNmodel_2/conv2d_14/mul_1:z:0"model_2/conv2d_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517208*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@h
model_2/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
model_2/flatten_4/ReshapeReshape$model_2/conv2d_14/IdentityN:output:0 model_2/flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
model_2/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
model_2/flatten_5/ReshapeReshape#model_2/dense_13/IdentityN:output:0 model_2/flatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ø
model_2/concatenate_2/concatConcatV2"model_2/flatten_4/Reshape:output:0"model_2/flatten_5/Reshape:output:0*model_2/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&model_2/dense_14/MatMul/ReadVariableOpReadVariableOp/model_2_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0«
model_2/dense_14/MatMulMatMul%model_2/concatenate_2/concat:output:0.model_2/dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model_2/dense_14/BiasAddBiasAdd!model_2/dense_14/MatMul:product:0/model_2/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
model_2/dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_14/mulMulmodel_2/dense_14/beta:output:0!model_2/dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model_2/dense_14/SigmoidSigmoidmodel_2/dense_14/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_2/dense_14/mul_1Mul!model_2/dense_14/BiasAdd:output:0model_2/dense_14/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
model_2/dense_14/IdentityIdentitymodel_2/dense_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
model_2/dense_14/IdentityN	IdentityNmodel_2/dense_14/mul_1:z:0!model_2/dense_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517228*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
&model_2/dense_15/MatMul/ReadVariableOpReadVariableOp/model_2_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
model_2/dense_15/MatMulMatMul#model_2/dense_14/IdentityN:output:0.model_2/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/dense_15/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model_2/dense_15/BiasAddBiasAdd!model_2/dense_15/MatMul:product:0/model_2/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
model_2/dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_15/mulMulmodel_2/dense_15/beta:output:0!model_2/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model_2/dense_15/SigmoidSigmoidmodel_2/dense_15/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_2/dense_15/mul_1Mul!model_2/dense_15/BiasAdd:output:0model_2/dense_15/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
model_2/dense_15/IdentityIdentitymodel_2/dense_15/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
model_2/dense_15/IdentityN	IdentityNmodel_2/dense_15/mul_1:z:0!model_2/dense_15/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517242*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
&model_2/dense_16/MatMul/ReadVariableOpReadVariableOp/model_2_dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¨
model_2/dense_16/MatMulMatMul#model_2/dense_15/IdentityN:output:0.model_2/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
'model_2/dense_16/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
model_2/dense_16/BiasAddBiasAdd!model_2/dense_16/MatMul:product:0/model_2/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
model_2/dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_2/dense_16/mulMulmodel_2/dense_16/beta:output:0!model_2/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
model_2/dense_16/SigmoidSigmoidmodel_2/dense_16/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model_2/dense_16/mul_1Mul!model_2/dense_16/BiasAdd:output:0model_2/dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
model_2/dense_16/IdentityIdentitymodel_2/dense_16/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ß
model_2/dense_16/IdentityN	IdentityNmodel_2/dense_16/mul_1:z:0!model_2/dense_16/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517256*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
&model_2/dense_17/MatMul/ReadVariableOpReadVariableOp/model_2_dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¨
model_2/dense_17/MatMulMatMul#model_2/dense_16/IdentityN:output:0.model_2/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/dense_17/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model_2/dense_17/BiasAddBiasAdd!model_2/dense_17/MatMul:product:0/model_2/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
IdentityIdentity!model_2/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
NoOpNoOp)^model_2/conv2d_10/BiasAdd/ReadVariableOp(^model_2/conv2d_10/Conv2D/ReadVariableOp)^model_2/conv2d_11/BiasAdd/ReadVariableOp(^model_2/conv2d_11/Conv2D/ReadVariableOp)^model_2/conv2d_12/BiasAdd/ReadVariableOp(^model_2/conv2d_12/Conv2D/ReadVariableOp)^model_2/conv2d_13/BiasAdd/ReadVariableOp(^model_2/conv2d_13/Conv2D/ReadVariableOp)^model_2/conv2d_14/BiasAdd/ReadVariableOp(^model_2/conv2d_14/Conv2D/ReadVariableOp(^model_2/dense_12/BiasAdd/ReadVariableOp'^model_2/dense_12/MatMul/ReadVariableOp(^model_2/dense_13/BiasAdd/ReadVariableOp'^model_2/dense_13/MatMul/ReadVariableOp(^model_2/dense_14/BiasAdd/ReadVariableOp'^model_2/dense_14/MatMul/ReadVariableOp(^model_2/dense_15/BiasAdd/ReadVariableOp'^model_2/dense_15/MatMul/ReadVariableOp(^model_2/dense_16/BiasAdd/ReadVariableOp'^model_2/dense_16/MatMul/ReadVariableOp(^model_2/dense_17/BiasAdd/ReadVariableOp'^model_2/dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2T
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
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
­
|
%__inference_internal_grad_fn_96519295
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Õ

%__inference_internal_grad_fn_96519439
result_grads_0
result_grads_1
mul_dense_13_beta
mul_dense_13_biasadd
identityv
mulMulmul_dense_13_betamul_dense_13_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
mul_1Mulmul_dense_13_betamul_dense_13_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :ÿÿÿÿÿÿÿÿÿ :W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Ï

G__inference_conv2d_11_layer_call_and_return_conditional_losses_96518632

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		¾
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518624*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
±
ú
F__inference_dense_16_layer_call_and_return_conditional_losses_96518903

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518895*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

G__inference_conv2d_12_layer_call_and_return_conditional_losses_96518669

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518661*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿl

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä

%__inference_internal_grad_fn_96519475
result_grads_0
result_grads_1
mul_dense_14_beta
mul_dense_14_biasadd
identityw
mulMulmul_dense_14_betamul_dense_14_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
mul_1Mulmul_dense_14_betamul_dense_14_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¾

G__inference_conv2d_14_layer_call_and_return_conditional_losses_96518760

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518752*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾

G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517459*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


%__inference_internal_grad_fn_96519115
result_grads_0
result_grads_1
mul_model_2_dense_14_beta 
mul_model_2_dense_14_biasadd
identity
mulMulmul_model_2_dense_14_betamul_model_2_dense_14_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
mul_1Mulmul_model_2_dense_14_betamul_model_2_dense_14_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_96518809

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ã

%__inference_internal_grad_fn_96519349
result_grads_0
result_grads_1
mul_conv2d_10_beta
mul_conv2d_10_biasadd
identity
mulMulmul_conv2d_10_betamul_conv2d_10_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		r
mul_1Mulmul_conv2d_10_betamul_conv2d_10_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
¥
|
%__inference_internal_grad_fn_96519187
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
Â

G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394

inputs9
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517386*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ü
F__inference_dense_15_layer_call_and_return_conditional_losses_96517540

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517532*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
£
,__inference_conv2d_10_layer_call_fn_96518587

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ		: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
Ë

+__inference_dense_12_layer_call_fn_96518715

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_96517418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ü
F__inference_dense_14_layer_call_and_return_conditional_losses_96518849

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518841*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
|
%__inference_internal_grad_fn_96519835
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

|
%__inference_internal_grad_fn_96519817
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :ÿÿÿÿÿÿÿÿÿ :W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
É	
÷
F__inference_dense_17_layer_call_and_return_conditional_losses_96517580

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

|
%__inference_internal_grad_fn_96519223
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
ã

%__inference_internal_grad_fn_96519529
result_grads_0
result_grads_1
mul_conv2d_10_beta
mul_conv2d_10_biasadd
identity
mulMulmul_conv2d_10_betamul_conv2d_10_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		r
mul_1Mulmul_conv2d_10_betamul_conv2d_10_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
Ë

G__inference_conv2d_10_layer_call_and_return_conditional_losses_96518605

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		¾
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518597*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs

|
%__inference_internal_grad_fn_96519241
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
ã

%__inference_internal_grad_fn_96519367
result_grads_0
result_grads_1
mul_conv2d_11_beta
mul_conv2d_11_biasadd
identity
mulMulmul_conv2d_11_betamul_conv2d_11_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		r
mul_1Mulmul_conv2d_11_betamul_conv2d_11_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
ä

%__inference_internal_grad_fn_96519673
result_grads_0
result_grads_1
mul_dense_15_beta
mul_dense_15_biasadd
identityw
mulMulmul_dense_15_betamul_dense_15_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
mul_1Mulmul_dense_15_betamul_dense_15_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

|
%__inference_internal_grad_fn_96519763
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
È
w
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96518822
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
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
3
	
!__inference__traced_save_96519912
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

identity_1¢MergeV2Checkpointsw
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
: Æ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ï	
valueå	Bâ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
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
_input_shapesï
ì: :::::::@:@:@:@:@@:@:@ : :
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
¾
ü
F__inference_dense_15_layer_call_and_return_conditional_losses_96518876

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mulMulbeta:output:0BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518868*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentityN:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´J
¼

E__inference_model_2_layer_call_and_return_conditional_losses_96518043
input_5
input_6-
conv2d_10_96517982:!
conv2d_10_96517984:	.
conv2d_11_96517987:!
conv2d_11_96517989:	.
conv2d_12_96517993:!
conv2d_12_96517995:	-
conv2d_13_96517998:@ 
conv2d_13_96518000:@#
dense_12_96518003:@
dense_12_96518005:@#
dense_13_96518009:@ 
dense_13_96518011: ,
conv2d_14_96518014:@@ 
conv2d_14_96518016:@%
dense_14_96518022:
  
dense_14_96518024:	%
dense_15_96518027:
 
dense_15_96518029:	$
dense_16_96518032:	@
dense_16_96518034:@#
dense_17_96518037:@
dense_17_96518039:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_96517982conv2d_10_96517984*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321¬
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_96517987conv2d_11_96517989*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345ú
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96517279ª
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_96517993conv2d_12_96517995*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370«
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_96517998conv2d_13_96518000*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394ü
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_96518003dense_12_96518005*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_96517418ù
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96517291
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_96518009dense_13_96518011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_96517443©
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_96518014conv2d_14_96518016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467æ
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_96517479ä
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_96517487
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96517496
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_96518022dense_14_96518024*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_96517516
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_96518027dense_15_96518029*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_96517540
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_96518032dense_16_96518034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_96517564
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_96518037dense_17_96518039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_96517580x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2F
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
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
²J
¼

E__inference_model_2_layer_call_and_return_conditional_losses_96517881

inputs
inputs_1-
conv2d_10_96517820:!
conv2d_10_96517822:	.
conv2d_11_96517825:!
conv2d_11_96517827:	.
conv2d_12_96517831:!
conv2d_12_96517833:	-
conv2d_13_96517836:@ 
conv2d_13_96517838:@#
dense_12_96517841:@
dense_12_96517843:@#
dense_13_96517847:@ 
dense_13_96517849: ,
conv2d_14_96517852:@@ 
conv2d_14_96517854:@%
dense_14_96517860:
  
dense_14_96517862:	%
dense_15_96517865:
 
dense_15_96517867:	$
dense_16_96517870:	@
dense_16_96517872:@#
dense_17_96517875:@
dense_17_96517877:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_96517820conv2d_10_96517822*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321¬
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_96517825conv2d_11_96517827*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345ú
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96517279ª
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_96517831conv2d_12_96517833*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370«
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_96517836conv2d_13_96517838*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394ý
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_96517841dense_12_96517843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_96517418ù
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96517291
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_96517847dense_13_96517849*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_96517443©
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_96517852conv2d_14_96517854*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467æ
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_96517479ä
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_96517487
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96517496
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_96517860dense_14_96517862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_96517516
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_96517865dense_15_96517867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_96517540
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_96517870dense_16_96517872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_96517564
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_96517875dense_17_96517877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_96517580x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2F
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
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¤
,__inference_conv2d_11_layer_call_fn_96518614

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ		: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
ä

%__inference_internal_grad_fn_96519493
result_grads_0
result_grads_1
mul_dense_15_beta
mul_dense_15_biasadd
identityw
mulMulmul_dense_15_betamul_dense_15_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
mul_1Mulmul_dense_15_betamul_dense_15_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

|
%__inference_internal_grad_fn_96519871
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Â

G__inference_conv2d_13_layer_call_and_return_conditional_losses_96518696

inputs9
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
mulMulbeta:output:0BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518688*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@k

Identity_1IdentityIdentityN:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

%__inference_internal_grad_fn_96519637
result_grads_0
result_grads_1
mul_conv2d_14_beta
mul_conv2d_14_biasadd
identity
mulMulmul_conv2d_14_betamul_conv2d_14_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
mul_1Mulmul_conv2d_14_betamul_conv2d_14_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

|
%__inference_internal_grad_fn_96519259
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityd
mulMulmul_betamul_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
mul_1Mulmul_betamul_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ : :ÿÿÿÿÿÿÿÿÿ :W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
Î

+__inference_dense_16_layer_call_fn_96518885

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_96517564o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
H
,__inference_flatten_5_layer_call_fn_96518803

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_96517487`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96517291

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
|
%__inference_internal_grad_fn_96519745
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
²J
¼

E__inference_model_2_layer_call_and_return_conditional_losses_96517587

inputs
inputs_1-
conv2d_10_96517322:!
conv2d_10_96517324:	.
conv2d_11_96517346:!
conv2d_11_96517348:	.
conv2d_12_96517371:!
conv2d_12_96517373:	-
conv2d_13_96517395:@ 
conv2d_13_96517397:@#
dense_12_96517419:@
dense_12_96517421:@#
dense_13_96517444:@ 
dense_13_96517446: ,
conv2d_14_96517468:@@ 
conv2d_14_96517470:@%
dense_14_96517517:
  
dense_14_96517519:	%
dense_15_96517541:
 
dense_15_96517543:	$
dense_16_96517565:	@
dense_16_96517567:@#
dense_17_96517581:@
dense_17_96517583:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_96517322conv2d_10_96517324*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321¬
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_96517346conv2d_11_96517348*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345ú
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96517279ª
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_96517371conv2d_12_96517373*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370«
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_96517395conv2d_13_96517397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394ý
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_96517419dense_12_96517421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_96517418ù
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96517291
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_96517444dense_13_96517446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_96517443©
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_96517468conv2d_14_96517470*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467æ
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_96517479ä
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_96517487
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96517496
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_96517517dense_14_96517519*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_96517516
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_96517541dense_15_96517543*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_96517540
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_96517565dense_16_96517567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_96517564
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_96517581dense_17_96517583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_96517580x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2F
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
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
ù
F__inference_dense_12_layer_call_and_return_conditional_losses_96517418

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517410*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
|
%__inference_internal_grad_fn_96519169
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
¥
|
%__inference_internal_grad_fn_96519727
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
Ò

+__inference_dense_15_layer_call_fn_96518858

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_96517540p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321

inputs9
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		f
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		¾
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517313*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		l

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
Ô

%__inference_internal_grad_fn_96519403
result_grads_0
result_grads_1
mul_conv2d_13_beta
mul_conv2d_13_biasadd
identity
mulMulmul_conv2d_13_betamul_conv2d_13_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
mul_1Mulmul_conv2d_13_betamul_conv2d_13_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Õ

%__inference_internal_grad_fn_96519421
result_grads_0
result_grads_1
mul_dense_12_beta
mul_dense_12_biasadd
identityv
mulMulmul_dense_12_betamul_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
mul_1Mulmul_dense_12_betamul_dense_12_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
ã

%__inference_internal_grad_fn_96519547
result_grads_0
result_grads_1
mul_conv2d_11_beta
mul_conv2d_11_biasadd
identity
mulMulmul_conv2d_11_betamul_conv2d_11_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		r
mul_1Mulmul_conv2d_11_betamul_conv2d_11_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
Ò

+__inference_dense_14_layer_call_fn_96518831

inputs
unknown:
 
	unknown_0:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_96517516p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Á
N
2__inference_max_pooling2d_5_layer_call_fn_96518701

inputs
identityà
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96517291
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

 
%__inference_internal_grad_fn_96519025
result_grads_0
result_grads_1
mul_model_2_conv2d_12_beta!
mul_model_2_conv2d_12_biasadd
identity
mulMulmul_model_2_conv2d_12_betamul_model_2_conv2d_12_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
mul_1Mulmul_model_2_conv2d_12_betamul_model_2_conv2d_12_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

|
%__inference_internal_grad_fn_96519277
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identityl
mulMulmul_betamul_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1Mulmul_betamul_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
·
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_96517487

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


%__inference_internal_grad_fn_96519061
result_grads_0
result_grads_1
mul_model_2_dense_12_beta 
mul_model_2_dense_12_biasadd
identity
mulMulmul_model_2_dense_12_betamul_model_2_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
mul_1Mulmul_model_2_dense_12_betamul_model_2_dense_12_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
ã

%__inference_internal_grad_fn_96519565
result_grads_0
result_grads_1
mul_conv2d_12_beta
mul_conv2d_12_biasadd
identity
mulMulmul_conv2d_12_betamul_conv2d_12_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
mul_1Mulmul_conv2d_12_betamul_conv2d_12_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¥
|
%__inference_internal_grad_fn_96519709
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitym
mulMulmul_betamul_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		^
mul_1Mulmul_betamul_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
±
ú
F__inference_dense_16_layer_call_and_return_conditional_losses_96517564

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¬
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517556*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
|
%__inference_internal_grad_fn_96519313
result_grads_0
result_grads_1
mul_beta
mul_biasadd
identitye
mulMulmul_betamul_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1Mulmul_betamul_biasadd*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_2Mul	mul_1:z:0sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
addAddV2add/x:output:0	mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_3MulSigmoid:y:0add:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_4Mulresult_grads_0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
IdentityIdentity	mul_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
%__inference_internal_grad_fn_96518989
result_grads_0
result_grads_1
mul_model_2_conv2d_10_beta!
mul_model_2_conv2d_10_biasadd
identity
mulMulmul_model_2_conv2d_10_betamul_model_2_conv2d_10_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
mul_1Mulmul_model_2_conv2d_10_betamul_model_2_conv2d_10_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		b
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Z
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		: :ÿÿÿÿÿÿÿÿÿ		:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
¹
\
0__inference_concatenate_2_layer_call_fn_96518815
inputs_0
inputs_1
identityÉ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96517496a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
Õ

%__inference_internal_grad_fn_96519601
result_grads_0
result_grads_1
mul_dense_12_beta
mul_dense_12_biasadd
identityv
mulMulmul_dense_12_betamul_dense_12_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
mul_1Mulmul_dense_12_betamul_dense_12_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Ë

+__inference_dense_13_layer_call_fn_96518769

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_96517443o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

 
%__inference_internal_grad_fn_96519097
result_grads_0
result_grads_1
mul_model_2_conv2d_14_beta!
mul_model_2_conv2d_14_biasadd
identity
mulMulmul_model_2_conv2d_14_betamul_model_2_conv2d_14_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
mul_1Mulmul_model_2_conv2d_14_betamul_model_2_conv2d_14_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
¡X
Ñ
$__inference__traced_restore_96519988
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
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9É

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ï	
valueå	Bâ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
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
 ³
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
­
ù
F__inference_dense_13_layer_call_and_return_conditional_losses_96517443

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 

identity_1¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
mulMulbeta:output:0BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517435*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ c

Identity_1IdentityIdentityN:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

 
%__inference_internal_grad_fn_96519043
result_grads_0
result_grads_1
mul_model_2_conv2d_13_beta!
mul_model_2_conv2d_13_biasadd
identity
mulMulmul_model_2_conv2d_13_betamul_model_2_conv2d_13_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
mul_1Mulmul_model_2_conv2d_13_betamul_model_2_conv2d_13_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
É
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_96518798

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î

*__inference_model_2_layer_call_fn_96517978
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
identity¢StatefulPartitionedCallù
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
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_96517881o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
ô

*__inference_model_2_layer_call_fn_96518208
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
identity¢StatefulPartitionedCallû
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
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_96517881o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

i
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96518706

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	

identity_1¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?f
mulMulbeta:output:0BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96517362*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿl

Identity_1IdentityIdentityN:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
¢
,__inference_conv2d_13_layer_call_fn_96518678

inputs"
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_17_layer_call_and_return_conditional_losses_96518922

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
´J
¼

E__inference_model_2_layer_call_and_return_conditional_losses_96518108
input_5
input_6-
conv2d_10_96518047:!
conv2d_10_96518049:	.
conv2d_11_96518052:!
conv2d_11_96518054:	.
conv2d_12_96518058:!
conv2d_12_96518060:	-
conv2d_13_96518063:@ 
conv2d_13_96518065:@#
dense_12_96518068:@
dense_12_96518070:@#
dense_13_96518074:@ 
dense_13_96518076: ,
conv2d_14_96518079:@@ 
conv2d_14_96518081:@%
dense_14_96518087:
  
dense_14_96518089:	%
dense_15_96518092:
 
dense_15_96518094:	$
dense_16_96518097:	@
dense_16_96518099:@#
dense_17_96518102:@
dense_17_96518104:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_96518047conv2d_10_96518049*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321¬
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_96518052conv2d_11_96518054*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345ú
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96517279ª
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_96518058conv2d_12_96518060*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370«
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_96518063conv2d_13_96518065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394ü
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_96518068dense_12_96518070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_96517418ù
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96517291
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_96518074dense_13_96518076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_96517443©
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_96518079conv2d_14_96518081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467æ
flatten_4/PartitionedCallPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_96517479ä
flatten_5/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_96517487
concatenate_2/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96517496
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_96518087dense_14_96518089*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_96517516
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_96518092dense_15_96518094*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_96517540
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_96518097dense_16_96518099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_16_layer_call_and_return_conditional_losses_96517564
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_96518102dense_17_96518104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_dense_17_layer_call_and_return_conditional_losses_96517580x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2F
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
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6

i
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96517279

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

%__inference_internal_grad_fn_96519583
result_grads_0
result_grads_1
mul_conv2d_13_beta
mul_conv2d_13_biasadd
identity
mulMulmul_conv2d_13_betamul_conv2d_13_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
mul_1Mulmul_conv2d_13_betamul_conv2d_13_biasadd*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
mul_2Mul	mul_1:z:0sub:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/x:output:0	mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\
mul_3MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_4Mulresult_grads_0	mul_3:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentity	mul_4:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
ü
¤
,__inference_conv2d_12_layer_call_fn_96518651

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

E__inference_model_2_layer_call_and_return_conditional_losses_96518526
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
identity¢ conv2d_10/BiasAdd/ReadVariableOp¢conv2d_10/Conv2D/ReadVariableOp¢ conv2d_11/BiasAdd/ReadVariableOp¢conv2d_11/Conv2D/ReadVariableOp¢ conv2d_12/BiasAdd/ReadVariableOp¢conv2d_12/Conv2D/ReadVariableOp¢ conv2d_13/BiasAdd/ReadVariableOp¢conv2d_13/Conv2D/ReadVariableOp¢ conv2d_14/BiasAdd/ReadVariableOp¢conv2d_14/Conv2D/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢dense_14/MatMul/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0°
conv2d_10/Conv2DConv2Dinputs_0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		S
conv2d_10/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_10/mulMulconv2d_10/beta:output:0conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		j
conv2d_10/SigmoidSigmoidconv2d_10/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
conv2d_10/mul_1Mulconv2d_10/BiasAdd:output:0conv2d_10/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		n
conv2d_10/IdentityIdentityconv2d_10/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ü
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518378*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
conv2d_11/Conv2DConv2Dconv2d_10/IdentityN:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
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
:ÿÿÿÿÿÿÿÿÿ		S
conv2d_11/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_11/mulMulconv2d_11/beta:output:0conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		j
conv2d_11/SigmoidSigmoidconv2d_11/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
conv2d_11/mul_1Mulconv2d_11/BiasAdd:output:0conv2d_11/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		n
conv2d_11/IdentityIdentityconv2d_11/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ü
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518392*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		®
max_pooling2d_4/MaxPoolMaxPoolconv2d_11/IdentityN:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0È
conv2d_12/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿS
conv2d_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_12/mulMulconv2d_12/beta:output:0conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
conv2d_12/SigmoidSigmoidconv2d_12/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_12/mul_1Mulconv2d_12/BiasAdd:output:0conv2d_12/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
conv2d_12/IdentityIdentityconv2d_12/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518407*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ã
conv2d_13/Conv2DConv2Dconv2d_12/IdentityN:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@S
conv2d_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_13/mulMulconv2d_13/beta:output:0conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
conv2d_13/SigmoidSigmoidconv2d_13/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_13/mul_1Mulconv2d_13/BiasAdd:output:0conv2d_13/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
conv2d_13/IdentityIdentityconv2d_13/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518421*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0}
dense_12/MatMulMatMulinputs_1&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
dense_12/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_12/mulMuldense_12/beta:output:0dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
dense_12/SigmoidSigmoiddense_12/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
dense_12/mul_1Muldense_12/BiasAdd:output:0dense_12/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
dense_12/IdentityIdentitydense_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518435*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d_5/MaxPoolMaxPoolconv2d_13/IdentityN:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ 
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ R
dense_13/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_13/mulMuldense_13/beta:output:0dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
dense_13/SigmoidSigmoiddense_13/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ x
dense_13/mul_1Muldense_13/BiasAdd:output:0dense_13/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c
dense_13/IdentityIdentitydense_13/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ç
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518450*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ 
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ç
conv2d_14/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@S
conv2d_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_14/mulMulconv2d_14/beta:output:0conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
conv2d_14/SigmoidSigmoidconv2d_14/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_14/mul_1Mulconv2d_14/BiasAdd:output:0conv2d_14/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
conv2d_14/IdentityIdentityconv2d_14/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ú
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518464*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_4/ReshapeReshapeconv2d_14/IdentityN:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_5/ReshapeReshapedense_13/IdentityN:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ [
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¸
concatenate_2/concatConcatV2flatten_4/Reshape:output:0flatten_5/Reshape:output:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_14/MatMulMatMulconcatenate_2/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dense_14/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_14/mulMuldense_14/beta:output:0dense_14/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_14/SigmoidSigmoiddense_14/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
dense_14/mul_1Muldense_14/BiasAdd:output:0dense_14/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/IdentityIdentitydense_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518484*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_15/MatMulMatMuldense_14/IdentityN:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dense_15/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
dense_15/mulMuldense_15/beta:output:0dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_15/SigmoidSigmoiddense_15/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
dense_15/mul_1Muldense_15/BiasAdd:output:0dense_15/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_15/IdentityIdentitydense_15/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518498*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_16/MatMulMatMuldense_15/IdentityN:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
dense_16/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
dense_16/mulMuldense_16/beta:output:0dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
dense_16/SigmoidSigmoiddense_16/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@x
dense_16/mul_1Muldense_16/BiasAdd:output:0dense_16/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c
dense_16/IdentityIdentitydense_16/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-96518512*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_17/MatMulMatMuldense_16/IdentityN:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2D
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
:ÿÿÿÿÿÿÿÿÿ		
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Õ

%__inference_internal_grad_fn_96519511
result_grads_0
result_grads_1
mul_dense_16_beta
mul_dense_16_biasadd
identityv
mulMulmul_dense_16_betamul_dense_16_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
mul_1Mulmul_dense_16_betamul_dense_16_biasadd*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
addAddV2add/x:output:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
mul_4Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
IdentityIdentity	mul_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@: :ÿÿÿÿÿÿÿÿÿ@:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_0:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameresult_grads_1:

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
Á
N
2__inference_max_pooling2d_4_layer_call_fn_96518637

inputs
identityà
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *V
fQRO
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96517279
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î

*__inference_model_2_layer_call_fn_96517634
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
identity¢StatefulPartitionedCallù
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
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_model_2_layer_call_and_return_conditional_losses_96517587o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
È

&__inference_signature_wrapper_96518578
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
identity¢StatefulPartitionedCall×
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
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *,
f'R%
#__inference__wrapped_model_96517270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_5:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_6
ã

%__inference_internal_grad_fn_96519385
result_grads_0
result_grads_1
mul_conv2d_12_beta
mul_conv2d_12_biasadd
identity
mulMulmul_conv2d_12_betamul_conv2d_12_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
mul_1Mulmul_conv2d_12_betamul_conv2d_12_biasadd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mul	mul_1:z:0sub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?b
addAddV2add/x:output:0	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_3MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
mul_4Mulresult_grads_0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity	mul_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*i
_input_shapesX
V:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :ÿÿÿÿÿÿÿÿÿ:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameresult_grads_1:

_output_shapes
: :62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%__inference_internal_grad_fn_96518989CustomGradient-96517122@
%__inference_internal_grad_fn_96519007CustomGradient-96517136@
%__inference_internal_grad_fn_96519025CustomGradient-96517151@
%__inference_internal_grad_fn_96519043CustomGradient-96517165@
%__inference_internal_grad_fn_96519061CustomGradient-96517179@
%__inference_internal_grad_fn_96519079CustomGradient-96517194@
%__inference_internal_grad_fn_96519097CustomGradient-96517208@
%__inference_internal_grad_fn_96519115CustomGradient-96517228@
%__inference_internal_grad_fn_96519133CustomGradient-96517242@
%__inference_internal_grad_fn_96519151CustomGradient-96517256@
%__inference_internal_grad_fn_96519169CustomGradient-96517313@
%__inference_internal_grad_fn_96519187CustomGradient-96517337@
%__inference_internal_grad_fn_96519205CustomGradient-96517362@
%__inference_internal_grad_fn_96519223CustomGradient-96517386@
%__inference_internal_grad_fn_96519241CustomGradient-96517410@
%__inference_internal_grad_fn_96519259CustomGradient-96517435@
%__inference_internal_grad_fn_96519277CustomGradient-96517459@
%__inference_internal_grad_fn_96519295CustomGradient-96517508@
%__inference_internal_grad_fn_96519313CustomGradient-96517532@
%__inference_internal_grad_fn_96519331CustomGradient-96517556@
%__inference_internal_grad_fn_96519349CustomGradient-96518219@
%__inference_internal_grad_fn_96519367CustomGradient-96518233@
%__inference_internal_grad_fn_96519385CustomGradient-96518248@
%__inference_internal_grad_fn_96519403CustomGradient-96518262@
%__inference_internal_grad_fn_96519421CustomGradient-96518276@
%__inference_internal_grad_fn_96519439CustomGradient-96518291@
%__inference_internal_grad_fn_96519457CustomGradient-96518305@
%__inference_internal_grad_fn_96519475CustomGradient-96518325@
%__inference_internal_grad_fn_96519493CustomGradient-96518339@
%__inference_internal_grad_fn_96519511CustomGradient-96518353@
%__inference_internal_grad_fn_96519529CustomGradient-96518378@
%__inference_internal_grad_fn_96519547CustomGradient-96518392@
%__inference_internal_grad_fn_96519565CustomGradient-96518407@
%__inference_internal_grad_fn_96519583CustomGradient-96518421@
%__inference_internal_grad_fn_96519601CustomGradient-96518435@
%__inference_internal_grad_fn_96519619CustomGradient-96518450@
%__inference_internal_grad_fn_96519637CustomGradient-96518464@
%__inference_internal_grad_fn_96519655CustomGradient-96518484@
%__inference_internal_grad_fn_96519673CustomGradient-96518498@
%__inference_internal_grad_fn_96519691CustomGradient-96518512@
%__inference_internal_grad_fn_96519709CustomGradient-96518597@
%__inference_internal_grad_fn_96519727CustomGradient-96518624@
%__inference_internal_grad_fn_96519745CustomGradient-96518661@
%__inference_internal_grad_fn_96519763CustomGradient-96518688@
%__inference_internal_grad_fn_96519781CustomGradient-96518725@
%__inference_internal_grad_fn_96519799CustomGradient-96518752@
%__inference_internal_grad_fn_96519817CustomGradient-96518779@
%__inference_internal_grad_fn_96519835CustomGradient-96518841@
%__inference_internal_grad_fn_96519853CustomGradient-96518868@
%__inference_internal_grad_fn_96519871CustomGradient-96518895"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ð
serving_defaultÜ
C
input_58
serving_default_input_5:0ÿÿÿÿÿÿÿÿÿ		
;
input_60
serving_default_input_6:0ÿÿÿÿÿÿÿÿÿ<
dense_170
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:×
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
à

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
à

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
Ê
#/_self_saveable_object_factories
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
à

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
à

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
Ê
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
à

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
à

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
à

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
Ê
#k_self_saveable_object_factories
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#r_self_saveable_object_factories
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#y_self_saveable_object_factories
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
é
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
é
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
é
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
é
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
 regularization_losses
¡	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"
_tf_keras_layer
-
¤serving_default"
signature_map
 "
trackable_dict_wrapper
Î
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
Î
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
Ï
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ö2ó
*__inference_model_2_layer_call_fn_96517634
*__inference_model_2_layer_call_fn_96518158
*__inference_model_2_layer_call_fn_96518208
*__inference_model_2_layer_call_fn_96517978À
·²³
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
kwonlydefaultsª 
annotationsª *
 
â2ß
E__inference_model_2_layer_call_and_return_conditional_losses_96518367
E__inference_model_2_layer_call_and_return_conditional_losses_96518526
E__inference_model_2_layer_call_and_return_conditional_losses_96518043
E__inference_model_2_layer_call_and_return_conditional_losses_96518108À
·²³
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
kwonlydefaultsª 
annotationsª *
 
×BÔ
#__inference__wrapped_model_96517270input_5input_6"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_10_layer_call_fn_96518587¢
²
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
annotationsª *
 
ñ2î
G__inference_conv2d_10_layer_call_and_return_conditional_losses_96518605¢
²
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
annotationsª *
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
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_11_layer_call_fn_96518614¢
²
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
annotationsª *
 
ñ2î
G__inference_conv2d_11_layer_call_and_return_conditional_losses_96518632¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_4_layer_call_fn_96518637¢
²
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
annotationsª *
 
÷2ô
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96518642¢
²
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
annotationsª *
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
²
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_12_layer_call_fn_96518651¢
²
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
annotationsª *
 
ñ2î
G__inference_conv2d_12_layer_call_and_return_conditional_losses_96518669¢
²
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
annotationsª *
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
²
¾non_trainable_variables
¿layers
Àmetrics
 Álayer_regularization_losses
Âlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_13_layer_call_fn_96518678¢
²
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
annotationsª *
 
ñ2î
G__inference_conv2d_13_layer_call_and_return_conditional_losses_96518696¢
²
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
annotationsª *
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
²
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_5_layer_call_fn_96518701¢
²
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
annotationsª *
 
÷2ô
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96518706¢
²
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
annotationsª *
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
²
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_12_layer_call_fn_96518715¢
²
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
annotationsª *
 
ð2í
F__inference_dense_12_layer_call_and_return_conditional_losses_96518733¢
²
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
annotationsª *
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
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_conv2d_14_layer_call_fn_96518742¢
²
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
annotationsª *
 
ñ2î
G__inference_conv2d_14_layer_call_and_return_conditional_losses_96518760¢
²
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
annotationsª *
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
²
Ònon_trainable_variables
Ólayers
Ômetrics
 Õlayer_regularization_losses
Ölayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_13_layer_call_fn_96518769¢
²
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
annotationsª *
 
ð2í
F__inference_dense_13_layer_call_and_return_conditional_losses_96518787¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_flatten_4_layer_call_fn_96518792¢
²
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
annotationsª *
 
ñ2î
G__inference_flatten_4_layer_call_and_return_conditional_losses_96518798¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ünon_trainable_variables
Ýlayers
Þmetrics
 ßlayer_regularization_losses
àlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_flatten_5_layer_call_fn_96518803¢
²
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
annotationsª *
 
ñ2î
G__inference_flatten_5_layer_call_and_return_conditional_losses_96518809¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ánon_trainable_variables
âlayers
ãmetrics
 älayer_regularization_losses
ålayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_concatenate_2_layer_call_fn_96518815¢
²
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
annotationsª *
 
õ2ò
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96518822¢
²
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
annotationsª *
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
¸
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_14_layer_call_fn_96518831¢
²
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
annotationsª *
 
ð2í
F__inference_dense_14_layer_call_and_return_conditional_losses_96518849¢
²
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
annotationsª *
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
¸
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_15_layer_call_fn_96518858¢
²
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
annotationsª *
 
ð2í
F__inference_dense_15_layer_call_and_return_conditional_losses_96518876¢
²
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
annotationsª *
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
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_16_layer_call_fn_96518885¢
²
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
annotationsª *
 
ð2í
F__inference_dense_16_layer_call_and_return_conditional_losses_96518903¢
²
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
annotationsª *
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
¸
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
	variables
trainable_variables
 regularization_losses
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_dense_17_layer_call_fn_96518912¢
²
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
annotationsª *
 
ð2í
F__inference_dense_17_layer_call_and_return_conditional_losses_96518922¢
²
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
annotationsª *
 
ÔBÑ
&__inference_signature_wrapper_96518578input_5input_6"
²
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
annotationsª *
 
 "
trackable_list_wrapper
¦
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
model_2/conv2d_10/beta:0#__inference__wrapped_model_96517270
DbB
model_2/conv2d_10/BiasAdd:0#__inference__wrapped_model_96517270
Ab?
model_2/conv2d_11/beta:0#__inference__wrapped_model_96517270
DbB
model_2/conv2d_11/BiasAdd:0#__inference__wrapped_model_96517270
Ab?
model_2/conv2d_12/beta:0#__inference__wrapped_model_96517270
DbB
model_2/conv2d_12/BiasAdd:0#__inference__wrapped_model_96517270
Ab?
model_2/conv2d_13/beta:0#__inference__wrapped_model_96517270
DbB
model_2/conv2d_13/BiasAdd:0#__inference__wrapped_model_96517270
@b>
model_2/dense_12/beta:0#__inference__wrapped_model_96517270
CbA
model_2/dense_12/BiasAdd:0#__inference__wrapped_model_96517270
@b>
model_2/dense_13/beta:0#__inference__wrapped_model_96517270
CbA
model_2/dense_13/BiasAdd:0#__inference__wrapped_model_96517270
Ab?
model_2/conv2d_14/beta:0#__inference__wrapped_model_96517270
DbB
model_2/conv2d_14/BiasAdd:0#__inference__wrapped_model_96517270
@b>
model_2/dense_14/beta:0#__inference__wrapped_model_96517270
CbA
model_2/dense_14/BiasAdd:0#__inference__wrapped_model_96517270
@b>
model_2/dense_15/beta:0#__inference__wrapped_model_96517270
CbA
model_2/dense_15/BiasAdd:0#__inference__wrapped_model_96517270
@b>
model_2/dense_16/beta:0#__inference__wrapped_model_96517270
CbA
model_2/dense_16/BiasAdd:0#__inference__wrapped_model_96517270
SbQ
beta:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321
VbT
	BiasAdd:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_96517321
SbQ
beta:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345
VbT
	BiasAdd:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_96517345
SbQ
beta:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370
VbT
	BiasAdd:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_96517370
SbQ
beta:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394
VbT
	BiasAdd:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_96517394
RbP
beta:0F__inference_dense_12_layer_call_and_return_conditional_losses_96517418
UbS
	BiasAdd:0F__inference_dense_12_layer_call_and_return_conditional_losses_96517418
RbP
beta:0F__inference_dense_13_layer_call_and_return_conditional_losses_96517443
UbS
	BiasAdd:0F__inference_dense_13_layer_call_and_return_conditional_losses_96517443
SbQ
beta:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467
VbT
	BiasAdd:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_96517467
RbP
beta:0F__inference_dense_14_layer_call_and_return_conditional_losses_96517516
UbS
	BiasAdd:0F__inference_dense_14_layer_call_and_return_conditional_losses_96517516
RbP
beta:0F__inference_dense_15_layer_call_and_return_conditional_losses_96517540
UbS
	BiasAdd:0F__inference_dense_15_layer_call_and_return_conditional_losses_96517540
RbP
beta:0F__inference_dense_16_layer_call_and_return_conditional_losses_96517564
UbS
	BiasAdd:0F__inference_dense_16_layer_call_and_return_conditional_losses_96517564
[bY
conv2d_10/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
^b\
conv2d_10/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
[bY
conv2d_11/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
^b\
conv2d_11/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
[bY
conv2d_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
^b\
conv2d_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
[bY
conv2d_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
^b\
conv2d_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
ZbX
dense_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
]b[
dense_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
ZbX
dense_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
]b[
dense_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
[bY
conv2d_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
^b\
conv2d_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
ZbX
dense_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
]b[
dense_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
ZbX
dense_15/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
]b[
dense_15/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
ZbX
dense_16/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
]b[
dense_16/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518367
[bY
conv2d_10/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
^b\
conv2d_10/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
[bY
conv2d_11/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
^b\
conv2d_11/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
[bY
conv2d_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
^b\
conv2d_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
[bY
conv2d_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
^b\
conv2d_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
ZbX
dense_12/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
]b[
dense_12/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
ZbX
dense_13/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
]b[
dense_13/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
[bY
conv2d_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
^b\
conv2d_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
ZbX
dense_14/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
]b[
dense_14/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
ZbX
dense_15/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
]b[
dense_15/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
ZbX
dense_16/beta:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
]b[
dense_16/BiasAdd:0E__inference_model_2_layer_call_and_return_conditional_losses_96518526
SbQ
beta:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_96518605
VbT
	BiasAdd:0G__inference_conv2d_10_layer_call_and_return_conditional_losses_96518605
SbQ
beta:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_96518632
VbT
	BiasAdd:0G__inference_conv2d_11_layer_call_and_return_conditional_losses_96518632
SbQ
beta:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_96518669
VbT
	BiasAdd:0G__inference_conv2d_12_layer_call_and_return_conditional_losses_96518669
SbQ
beta:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_96518696
VbT
	BiasAdd:0G__inference_conv2d_13_layer_call_and_return_conditional_losses_96518696
RbP
beta:0F__inference_dense_12_layer_call_and_return_conditional_losses_96518733
UbS
	BiasAdd:0F__inference_dense_12_layer_call_and_return_conditional_losses_96518733
SbQ
beta:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_96518760
VbT
	BiasAdd:0G__inference_conv2d_14_layer_call_and_return_conditional_losses_96518760
RbP
beta:0F__inference_dense_13_layer_call_and_return_conditional_losses_96518787
UbS
	BiasAdd:0F__inference_dense_13_layer_call_and_return_conditional_losses_96518787
RbP
beta:0F__inference_dense_14_layer_call_and_return_conditional_losses_96518849
UbS
	BiasAdd:0F__inference_dense_14_layer_call_and_return_conditional_losses_96518849
RbP
beta:0F__inference_dense_15_layer_call_and_return_conditional_losses_96518876
UbS
	BiasAdd:0F__inference_dense_15_layer_call_and_return_conditional_losses_96518876
RbP
beta:0F__inference_dense_16_layer_call_and_return_conditional_losses_96518903
UbS
	BiasAdd:0F__inference_dense_16_layer_call_and_return_conditional_losses_96518903ß
#__inference__wrapped_model_96517270·&'67?@PQbcYZ`¢]
V¢S
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿÕ
K__inference_concatenate_2_layer_call_and_return_conditional_losses_96518822[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 ¬
0__inference_concatenate_2_layer_call_fn_96518815x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¸
G__inference_conv2d_10_layer_call_and_return_conditional_losses_96518605m7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 
,__inference_conv2d_10_layer_call_fn_96518587`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª "!ÿÿÿÿÿÿÿÿÿ		¹
G__inference_conv2d_11_layer_call_and_return_conditional_losses_96518632n&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ		
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 
,__inference_conv2d_11_layer_call_fn_96518614a&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ		
ª "!ÿÿÿÿÿÿÿÿÿ		¹
G__inference_conv2d_12_layer_call_and_return_conditional_losses_96518669n678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_conv2d_12_layer_call_fn_96518651a678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
G__inference_conv2d_13_layer_call_and_return_conditional_losses_96518696m?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_13_layer_call_fn_96518678`?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@·
G__inference_conv2d_14_layer_call_and_return_conditional_losses_96518760lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_14_layer_call_fn_96518742_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dense_12_layer_call_and_return_conditional_losses_96518733\PQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_dense_12_layer_call_fn_96518715OPQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dense_13_layer_call_and_return_conditional_losses_96518787\bc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_dense_13_layer_call_fn_96518769Obc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ª
F__inference_dense_14_layer_call_and_return_conditional_losses_96518849`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_14_layer_call_fn_96518831S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿª
F__inference_dense_15_layer_call_and_return_conditional_losses_96518876`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_15_layer_call_fn_96518858S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
F__inference_dense_16_layer_call_and_return_conditional_losses_96518903_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_dense_16_layer_call_fn_96518885R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¨
F__inference_dense_17_layer_call_and_return_conditional_losses_96518922^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_17_layer_call_fn_96518912Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¬
G__inference_flatten_4_layer_call_and_return_conditional_losses_96518798a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_flatten_4_layer_call_fn_96518792T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ£
G__inference_flatten_5_layer_call_and_return_conditional_losses_96518809X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 {
,__inference_flatten_5_layer_call_fn_96518803K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ Ø
%__inference_internal_grad_fn_96518989®úûw¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519007®üýw¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519025®þÿw¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿÕ
%__inference_internal_grad_fn_96519043«u¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519061e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519079e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ 
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ 
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ Õ
%__inference_internal_grad_fn_96519097«u¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@À
%__inference_internal_grad_fn_96519115g¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿÀ
%__inference_internal_grad_fn_96519133g¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿ½
%__inference_internal_grad_fn_96519151e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@Ø
%__inference_internal_grad_fn_96519169®w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519187®w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519205®w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿÕ
%__inference_internal_grad_fn_96519223«u¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519241e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519259e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ 
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ 
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ Õ
%__inference_internal_grad_fn_96519277«u¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@À
%__inference_internal_grad_fn_96519295g¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿÀ
%__inference_internal_grad_fn_96519313g¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿ½
%__inference_internal_grad_fn_96519331 ¡e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@Ø
%__inference_internal_grad_fn_96519349®¢£w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519367®¤¥w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519385®¦§w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿÕ
%__inference_internal_grad_fn_96519403«¨©u¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519421ª«e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519439¬­e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ 
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ 
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ Õ
%__inference_internal_grad_fn_96519457«®¯u¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@À
%__inference_internal_grad_fn_96519475°±g¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿÀ
%__inference_internal_grad_fn_96519493²³g¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿ½
%__inference_internal_grad_fn_96519511´µe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@Ø
%__inference_internal_grad_fn_96519529®¶·w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519547®¸¹w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519565®º»w¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿÕ
%__inference_internal_grad_fn_96519583«¼½u¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519601¾¿e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519619ÀÁe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ 
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ 
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ Õ
%__inference_internal_grad_fn_96519637«ÂÃu¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@À
%__inference_internal_grad_fn_96519655ÄÅg¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿÀ
%__inference_internal_grad_fn_96519673ÆÇg¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿ½
%__inference_internal_grad_fn_96519691ÈÉe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@Ø
%__inference_internal_grad_fn_96519709®ÊËw¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519727®ÌÍw¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ		
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ		
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿ		Ø
%__inference_internal_grad_fn_96519745®ÎÏw¢t
m¢j

 
1.
result_grads_0ÿÿÿÿÿÿÿÿÿ
1.
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "-*

 
$!
1ÿÿÿÿÿÿÿÿÿÕ
%__inference_internal_grad_fn_96519763«ÐÑu¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519781ÒÓe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@Õ
%__inference_internal_grad_fn_96519799«ÔÕu¢r
k¢h

 
0-
result_grads_0ÿÿÿÿÿÿÿÿÿ@
0-
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª ",)

 
# 
1ÿÿÿÿÿÿÿÿÿ@½
%__inference_internal_grad_fn_96519817Ö×e¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ 
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ 
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ À
%__inference_internal_grad_fn_96519835ØÙg¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿÀ
%__inference_internal_grad_fn_96519853ÚÛg¢d
]¢Z

 
)&
result_grads_0ÿÿÿÿÿÿÿÿÿ
)&
result_grads_1ÿÿÿÿÿÿÿÿÿ
ª "%"

 

1ÿÿÿÿÿÿÿÿÿ½
%__inference_internal_grad_fn_96519871ÜÝe¢b
[¢X

 
(%
result_grads_0ÿÿÿÿÿÿÿÿÿ@
(%
result_grads_1ÿÿÿÿÿÿÿÿÿ@
ª "$!

 

1ÿÿÿÿÿÿÿÿÿ@ð
M__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_96518642R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_4_layer_call_fn_96518637R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_96518706R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_5_layer_call_fn_96518701R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿû
E__inference_model_2_layer_call_and_return_conditional_losses_96518043±&'67?@PQbcYZh¢e
^¢[
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 û
E__inference_model_2_layer_call_and_return_conditional_losses_96518108±&'67?@PQbcYZh¢e
^¢[
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ý
E__inference_model_2_layer_call_and_return_conditional_losses_96518367³&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ý
E__inference_model_2_layer_call_and_return_conditional_losses_96518526³&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ó
*__inference_model_2_layer_call_fn_96517634¤&'67?@PQbcYZh¢e
^¢[
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÓ
*__inference_model_2_layer_call_fn_96517978¤&'67?@PQbcYZh¢e
^¢[
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÕ
*__inference_model_2_layer_call_fn_96518158¦&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÕ
*__inference_model_2_layer_call_fn_96518208¦&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿó
&__inference_signature_wrapper_96518578È&'67?@PQbcYZq¢n
¢ 
gªd
4
input_5)&
input_5ÿÿÿÿÿÿÿÿÿ		
,
input_6!
input_6ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ