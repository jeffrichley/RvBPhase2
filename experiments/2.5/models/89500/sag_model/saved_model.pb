½¬
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68­ý
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
ê
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
GPU2*0,1J 8 *0
f+R)
'__inference_signature_wrapper_145817369
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Û
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
"__inference__traced_save_145818703
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
%__inference__traced_restore_145818779ÞÙ
¯
ú
G__inference_dense_13_layer_call_and_return_conditional_losses_145816234

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
:ÿÿÿÿÿÿÿÿÿ ­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816226*:
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
³
û
G__inference_dense_16_layer_call_and_return_conditional_losses_145817694

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
:ÿÿÿÿÿÿÿÿÿ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817686*:
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
Ê	
ø
G__inference_dense_17_layer_call_and_return_conditional_losses_145817713

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
Ä

H__inference_conv2d_13_layer_call_and_return_conditional_losses_145817487

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
:ÿÿÿÿÿÿÿÿÿ@½
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817479*J
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
¦
}
&__inference_internal_grad_fn_145817960
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
Ñ

H__inference_conv2d_11_layer_call_and_return_conditional_losses_145817423

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
:ÿÿÿÿÿÿÿÿÿ		¿
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817415*L
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
Ð

,__inference_dense_16_layer_call_fn_145817676

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallá
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_145816355o
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
¦
}
&__inference_internal_grad_fn_145817978
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


&__inference_internal_grad_fn_145817924
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
»
]
1__inference_concatenate_2_layer_call_fn_145817606
inputs_0
inputs_1
identityÊ
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
GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145816287a
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
Í

H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112

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
:ÿÿÿÿÿÿÿÿÿ		¿
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816104*L
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


&__inference_internal_grad_fn_145817870
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
Í

,__inference_dense_17_layer_call_fn_145817703

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallá
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_145816371o
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
Ê
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_145816270

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
Ê	
ø
G__inference_dense_17_layer_call_and_return_conditional_losses_145816371

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
®
}
&__inference_internal_grad_fn_145818086
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
Ô

,__inference_dense_14_layer_call_fn_145817622

inputs
unknown:
 
	unknown_0:	
identity¢StatefulPartitionedCallâ
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_145816307p
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
Õ

&__inference_internal_grad_fn_145818428
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

j
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145816070

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

¡
&__inference_internal_grad_fn_145817780
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


&__inference_internal_grad_fn_145817942
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
Õ

&__inference_internal_grad_fn_145818374
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

}
&__inference_internal_grad_fn_145818590
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
¦
}
&__inference_internal_grad_fn_145818536
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
É
x
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145817613
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
Ý

F__inference_model_2_layer_call_and_return_conditional_losses_145817317
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
:ÿÿÿÿÿÿÿÿÿ		Ý
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817169*L
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
:ÿÿÿÿÿÿÿÿÿ		Ý
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817183*L
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
:ÿÿÿÿÿÿÿÿÿÝ
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817198*L
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
:ÿÿÿÿÿÿÿÿÿ@Û
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817212*J
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
:ÿÿÿÿÿÿÿÿÿ@È
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817226*:
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
:ÿÿÿÿÿÿÿÿÿ È
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817241*:
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
:ÿÿÿÿÿÿÿÿÿ@Û
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817255*J
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
:ÿÿÿÿÿÿÿÿÿÊ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817275*<
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
:ÿÿÿÿÿÿÿÿÿÊ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817289*<
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
:ÿÿÿÿÿÿÿÿÿ@È
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817303*:
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
Ñ

H__inference_conv2d_12_layer_call_and_return_conditional_losses_145817460

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
:ÿÿÿÿÿÿÿÿÿ¿
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817452*L
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
Ê

'__inference_signature_wrapper_145817369
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
identity¢StatefulPartitionedCallØ
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
GPU2*0,1J 8 *-
f(R&
$__inference__wrapped_model_145816061o
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
À
ý
G__inference_dense_15_layer_call_and_return_conditional_losses_145817667

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
:ÿÿÿÿÿÿÿÿÿ¯
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817659*<
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
÷
¢
-__inference_conv2d_14_layer_call_fn_145817533

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallê
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258w
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
³
û
G__inference_dense_16_layer_call_and_return_conditional_losses_145816355

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
:ÿÿÿÿÿÿÿÿÿ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816347*:
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

}
&__inference_internal_grad_fn_145818608
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
Ã
O
3__inference_max_pooling2d_4_layer_call_fn_145817428

inputs
identityá
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145816070
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
¸
d
H__inference_flatten_5_layer_call_and_return_conditional_losses_145816278

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

}
&__inference_internal_grad_fn_145818122
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

}
&__inference_internal_grad_fn_145818068
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
À

H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258

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
:ÿÿÿÿÿÿÿÿÿ@½
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816250*J
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
ö

+__inference_model_2_layer_call_fn_145816999
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
identity¢StatefulPartitionedCallü
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
GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_145816672o
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
ä

&__inference_internal_grad_fn_145818338
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

¡
&__inference_internal_grad_fn_145817798
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
Í

,__inference_dense_13_layer_call_fn_145817560

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallá
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_145816234o
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

}
&__inference_internal_grad_fn_145818032
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

¡
&__inference_internal_grad_fn_145817816
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
À
ý
G__inference_dense_14_layer_call_and_return_conditional_losses_145816307

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
:ÿÿÿÿÿÿÿÿÿ¯
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816299*<
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
Ö

&__inference_internal_grad_fn_145818302
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
Í

,__inference_dense_12_layer_call_fn_145817506

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallá
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_145816209o
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
Õ

&__inference_internal_grad_fn_145818194
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
å

&__inference_internal_grad_fn_145818284
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
¶
Ï
$__inference__wrapped_model_145816061
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
:ÿÿÿÿÿÿÿÿÿ		õ
model_2/conv2d_10/IdentityN	IdentityNmodel_2/conv2d_10/mul_1:z:0"model_2/conv2d_10/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145815913*L
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
:ÿÿÿÿÿÿÿÿÿ		õ
model_2/conv2d_11/IdentityN	IdentityNmodel_2/conv2d_11/mul_1:z:0"model_2/conv2d_11/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145815927*L
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
:ÿÿÿÿÿÿÿÿÿõ
model_2/conv2d_12/IdentityN	IdentityNmodel_2/conv2d_12/mul_1:z:0"model_2/conv2d_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145815942*L
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
:ÿÿÿÿÿÿÿÿÿ@ó
model_2/conv2d_13/IdentityN	IdentityNmodel_2/conv2d_13/mul_1:z:0"model_2/conv2d_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145815956*J
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
:ÿÿÿÿÿÿÿÿÿ@à
model_2/dense_12/IdentityN	IdentityNmodel_2/dense_12/mul_1:z:0!model_2/dense_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145815970*:
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
:ÿÿÿÿÿÿÿÿÿ à
model_2/dense_13/IdentityN	IdentityNmodel_2/dense_13/mul_1:z:0!model_2/dense_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145815985*:
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
:ÿÿÿÿÿÿÿÿÿ@ó
model_2/conv2d_14/IdentityN	IdentityNmodel_2/conv2d_14/mul_1:z:0"model_2/conv2d_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145815999*J
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
:ÿÿÿÿÿÿÿÿÿâ
model_2/dense_14/IdentityN	IdentityNmodel_2/dense_14/mul_1:z:0!model_2/dense_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816019*<
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
:ÿÿÿÿÿÿÿÿÿâ
model_2/dense_15/IdentityN	IdentityNmodel_2/dense_15/mul_1:z:0!model_2/dense_15/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816033*<
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
:ÿÿÿÿÿÿÿÿÿ@à
model_2/dense_16/IdentityN	IdentityNmodel_2/dense_16/mul_1:z:0!model_2/dense_16/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816047*:
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
¸
d
H__inference_flatten_5_layer_call_and_return_conditional_losses_145817600

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

}
&__inference_internal_grad_fn_145818050
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
3
	
"__inference__traced_save_145818703
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

j
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145817497

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
ä

&__inference_internal_grad_fn_145818320
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
¯
ú
G__inference_dense_12_layer_call_and_return_conditional_losses_145817524

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
:ÿÿÿÿÿÿÿÿÿ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817516*:
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
ä

&__inference_internal_grad_fn_145818140
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
¦
}
&__inference_internal_grad_fn_145818500
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
û
¤
-__inference_conv2d_10_layer_call_fn_145817378

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallë
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112x
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
ª
I
-__inference_flatten_5_layer_call_fn_145817594

inputs
identity¸
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_145816278`
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
Ö

&__inference_internal_grad_fn_145818230
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
Ö

&__inference_internal_grad_fn_145818392
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

}
&__inference_internal_grad_fn_145818554
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
Ý

F__inference_model_2_layer_call_and_return_conditional_losses_145817158
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
:ÿÿÿÿÿÿÿÿÿ		Ý
conv2d_10/IdentityN	IdentityNconv2d_10/mul_1:z:0conv2d_10/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817010*L
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
:ÿÿÿÿÿÿÿÿÿ		Ý
conv2d_11/IdentityN	IdentityNconv2d_11/mul_1:z:0conv2d_11/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817024*L
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
:ÿÿÿÿÿÿÿÿÿÝ
conv2d_12/IdentityN	IdentityNconv2d_12/mul_1:z:0conv2d_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817039*L
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
:ÿÿÿÿÿÿÿÿÿ@Û
conv2d_13/IdentityN	IdentityNconv2d_13/mul_1:z:0conv2d_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817053*J
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
:ÿÿÿÿÿÿÿÿÿ@È
dense_12/IdentityN	IdentityNdense_12/mul_1:z:0dense_12/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817067*:
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
:ÿÿÿÿÿÿÿÿÿ È
dense_13/IdentityN	IdentityNdense_13/mul_1:z:0dense_13/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817082*:
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
:ÿÿÿÿÿÿÿÿÿ@Û
conv2d_14/IdentityN	IdentityNconv2d_14/mul_1:z:0conv2d_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817096*J
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
:ÿÿÿÿÿÿÿÿÿÊ
dense_14/IdentityN	IdentityNdense_14/mul_1:z:0dense_14/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817116*<
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
:ÿÿÿÿÿÿÿÿÿÊ
dense_15/IdentityN	IdentityNdense_15/mul_1:z:0dense_15/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817130*<
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
:ÿÿÿÿÿÿÿÿÿ@È
dense_16/IdentityN	IdentityNdense_16/mul_1:z:0dense_16/BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817144*:
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

j
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145816082

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
À
ý
G__inference_dense_15_layer_call_and_return_conditional_losses_145816331

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
:ÿÿÿÿÿÿÿÿÿ¯
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816323*<
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

j
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145817433

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
ïJ
Ó

F__inference_model_2_layer_call_and_return_conditional_losses_145816672

inputs
inputs_1.
conv2d_10_145816611:"
conv2d_10_145816613:	/
conv2d_11_145816616:"
conv2d_11_145816618:	/
conv2d_12_145816622:"
conv2d_12_145816624:	.
conv2d_13_145816627:@!
conv2d_13_145816629:@$
dense_12_145816632:@ 
dense_12_145816634:@$
dense_13_145816638:@  
dense_13_145816640: -
conv2d_14_145816643:@@!
conv2d_14_145816645:@&
dense_14_145816651:
 !
dense_14_145816653:	&
dense_15_145816656:
!
dense_15_145816658:	%
dense_16_145816661:	@ 
dense_16_145816663:@$
dense_17_145816666:@ 
dense_17_145816668:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_145816611conv2d_10_145816613*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112¯
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_145816616conv2d_11_145816618*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136û
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145816070­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_145816622conv2d_12_145816624*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161®
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_145816627conv2d_13_145816629*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_145816632dense_12_145816634*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_145816209ú
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145816082¡
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_145816638dense_13_145816640*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_145816234¬
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_145816643conv2d_14_145816645*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258ç
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_145816270å
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_145816278
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
GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145816287
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_145816651dense_14_145816653*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_145816307¢
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_145816656dense_15_145816658*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_145816331¡
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_145816661dense_16_145816663*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_145816355¡
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_145816666dense_17_145816668*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_145816371x
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
À
ý
G__inference_dense_14_layer_call_and_return_conditional_losses_145817640

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
:ÿÿÿÿÿÿÿÿÿ¯
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817632*<
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
Ñ

H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161

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
:ÿÿÿÿÿÿÿÿÿ¿
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816153*L
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
Á
v
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145816287

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
ð

+__inference_model_2_layer_call_fn_145816769
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
identity¢StatefulPartitionedCallú
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
GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_145816672o
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


&__inference_internal_grad_fn_145817852
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
ú
£
-__inference_conv2d_13_layer_call_fn_145817469

inputs"
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallê
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185w
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
ä

&__inference_internal_grad_fn_145818176
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
ä

&__inference_internal_grad_fn_145818158
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
Ñ

H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136

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
:ÿÿÿÿÿÿÿÿÿ		¿
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816128*L
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
Í

H__inference_conv2d_10_layer_call_and_return_conditional_losses_145817396

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
:ÿÿÿÿÿÿÿÿÿ		¿
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817388*L
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
¦
}
&__inference_internal_grad_fn_145818518
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
Ö

&__inference_internal_grad_fn_145818410
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

}
&__inference_internal_grad_fn_145818572
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
ïJ
Ó

F__inference_model_2_layer_call_and_return_conditional_losses_145816378

inputs
inputs_1.
conv2d_10_145816113:"
conv2d_10_145816115:	/
conv2d_11_145816137:"
conv2d_11_145816139:	/
conv2d_12_145816162:"
conv2d_12_145816164:	.
conv2d_13_145816186:@!
conv2d_13_145816188:@$
dense_12_145816210:@ 
dense_12_145816212:@$
dense_13_145816235:@  
dense_13_145816237: -
conv2d_14_145816259:@@!
conv2d_14_145816261:@&
dense_14_145816308:
 !
dense_14_145816310:	&
dense_15_145816332:
!
dense_15_145816334:	%
dense_16_145816356:	@ 
dense_16_145816358:@$
dense_17_145816372:@ 
dense_17_145816374:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_145816113conv2d_10_145816115*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112¯
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_145816137conv2d_11_145816139*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136û
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145816070­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_145816162conv2d_12_145816164*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161®
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_145816186conv2d_13_145816188*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_12_145816210dense_12_145816212*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_145816209ú
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145816082¡
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_145816235dense_13_145816237*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_145816234¬
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_145816259conv2d_14_145816261*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258ç
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_145816270å
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_145816278
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
GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145816287
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_145816308dense_14_145816310*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_145816307¢
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_145816332dense_15_145816334*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_145816331¡
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_145816356dense_16_145816358*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_145816355¡
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_145816372dense_17_145816374*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_145816371x
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
Ê
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_145817589

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
Ä

H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185

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
:ÿÿÿÿÿÿÿÿÿ@½
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816177*J
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
Ã
O
3__inference_max_pooling2d_5_layer_call_fn_145817492

inputs
identityá
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145816082
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
ñJ
Ó

F__inference_model_2_layer_call_and_return_conditional_losses_145816899
input_5
input_6.
conv2d_10_145816838:"
conv2d_10_145816840:	/
conv2d_11_145816843:"
conv2d_11_145816845:	/
conv2d_12_145816849:"
conv2d_12_145816851:	.
conv2d_13_145816854:@!
conv2d_13_145816856:@$
dense_12_145816859:@ 
dense_12_145816861:@$
dense_13_145816865:@  
dense_13_145816867: -
conv2d_14_145816870:@@!
conv2d_14_145816872:@&
dense_14_145816878:
 !
dense_14_145816880:	&
dense_15_145816883:
!
dense_15_145816885:	%
dense_16_145816888:	@ 
dense_16_145816890:@$
dense_17_145816893:@ 
dense_17_145816895:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_145816838conv2d_10_145816840*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112¯
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_145816843conv2d_11_145816845*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136û
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145816070­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_145816849conv2d_12_145816851*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161®
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_145816854conv2d_13_145816856*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185ÿ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_145816859dense_12_145816861*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_145816209ú
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145816082¡
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_145816865dense_13_145816867*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_145816234¬
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_145816870conv2d_14_145816872*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258ç
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_145816270å
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_145816278
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
GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145816287
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_145816878dense_14_145816880*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_145816307¢
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_145816883dense_15_145816885*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_145816331¡
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_145816888dense_16_145816890*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_145816355¡
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_145816893dense_17_145816895*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_145816371x
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
ö

+__inference_model_2_layer_call_fn_145816949
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
identity¢StatefulPartitionedCallü
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
GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_145816378o
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
¦
}
&__inference_internal_grad_fn_145817996
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

¡
&__inference_internal_grad_fn_145817888
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
å

&__inference_internal_grad_fn_145818266
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
¯
ú
G__inference_dense_13_layer_call_and_return_conditional_losses_145817578

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
:ÿÿÿÿÿÿÿÿÿ ­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817570*:
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
¯
ú
G__inference_dense_12_layer_call_and_return_conditional_losses_145816209

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
:ÿÿÿÿÿÿÿÿÿ@­
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145816201*:
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

¡
&__inference_internal_grad_fn_145817834
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
®
}
&__inference_internal_grad_fn_145818626
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
¼
I
-__inference_flatten_4_layer_call_fn_145817583

inputs
identity¹
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_145816270a
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
Ö

&__inference_internal_grad_fn_145818482
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
Õ

&__inference_internal_grad_fn_145818248
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
Ö

&__inference_internal_grad_fn_145818212
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
å

&__inference_internal_grad_fn_145818446
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
¢X
Ò
%__inference__traced_restore_145818779
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
Ô

,__inference_dense_15_layer_call_fn_145817649

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallâ
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_145816331p
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
®
}
&__inference_internal_grad_fn_145818104
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
þ
¥
-__inference_conv2d_12_layer_call_fn_145817442

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallë
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161x
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


&__inference_internal_grad_fn_145817906
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
®
}
&__inference_internal_grad_fn_145818644
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
À

H__inference_conv2d_14_layer_call_and_return_conditional_losses_145817551

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
:ÿÿÿÿÿÿÿÿÿ@½
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0*
T
2*/
_gradient_op_typeCustomGradient-145817543*J
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
å

&__inference_internal_grad_fn_145818464
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
ñJ
Ó

F__inference_model_2_layer_call_and_return_conditional_losses_145816834
input_5
input_6.
conv2d_10_145816773:"
conv2d_10_145816775:	/
conv2d_11_145816778:"
conv2d_11_145816780:	/
conv2d_12_145816784:"
conv2d_12_145816786:	.
conv2d_13_145816789:@!
conv2d_13_145816791:@$
dense_12_145816794:@ 
dense_12_145816796:@$
dense_13_145816800:@  
dense_13_145816802: -
conv2d_14_145816805:@@!
conv2d_14_145816807:@&
dense_14_145816813:
 !
dense_14_145816815:	&
dense_15_145816818:
!
dense_15_145816820:	%
dense_16_145816823:	@ 
dense_16_145816825:@$
dense_17_145816828:@ 
dense_17_145816830:
identity¢!conv2d_10/StatefulPartitionedCall¢!conv2d_11/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢!conv2d_13/StatefulPartitionedCall¢!conv2d_14/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_10_145816773conv2d_10_145816775*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112¯
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_145816778conv2d_11_145816780*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136û
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145816070­
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_12_145816784conv2d_12_145816786*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161®
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_145816789conv2d_13_145816791*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185ÿ
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_12_145816794dense_12_145816796*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_12_layer_call_and_return_conditional_losses_145816209ú
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
GPU2*0,1J 8 *W
fRRP
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145816082¡
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_145816800dense_13_145816802*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_13_layer_call_and_return_conditional_losses_145816234¬
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_14_145816805conv2d_14_145816807*
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258ç
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_145816270å
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_flatten_5_layer_call_and_return_conditional_losses_145816278
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
GPU2*0,1J 8 *U
fPRN
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145816287
 dense_14/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_14_145816813dense_14_145816815*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_14_layer_call_and_return_conditional_losses_145816307¢
 dense_15/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0dense_15_145816818dense_15_145816820*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_15_layer_call_and_return_conditional_losses_145816331¡
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_145816823dense_16_145816825*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_16_layer_call_and_return_conditional_losses_145816355¡
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_145816828dense_17_145816830*
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
GPU2*0,1J 8 *P
fKRI
G__inference_dense_17_layer_call_and_return_conditional_losses_145816371x
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
þ
¥
-__inference_conv2d_11_layer_call_fn_145817405

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallë
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
GPU2*0,1J 8 *Q
fLRJ
H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136x
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

}
&__inference_internal_grad_fn_145818014
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
ð

+__inference_model_2_layer_call_fn_145816425
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
identity¢StatefulPartitionedCallú
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
GPU2*0,1J 8 *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_145816378o
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
ä

&__inference_internal_grad_fn_145818356
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

}
&__inference_internal_grad_fn_145818662
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
:ÿÿÿÿÿÿÿÿÿ@B
&__inference_internal_grad_fn_145817780CustomGradient-145815913B
&__inference_internal_grad_fn_145817798CustomGradient-145815927B
&__inference_internal_grad_fn_145817816CustomGradient-145815942B
&__inference_internal_grad_fn_145817834CustomGradient-145815956B
&__inference_internal_grad_fn_145817852CustomGradient-145815970B
&__inference_internal_grad_fn_145817870CustomGradient-145815985B
&__inference_internal_grad_fn_145817888CustomGradient-145815999B
&__inference_internal_grad_fn_145817906CustomGradient-145816019B
&__inference_internal_grad_fn_145817924CustomGradient-145816033B
&__inference_internal_grad_fn_145817942CustomGradient-145816047B
&__inference_internal_grad_fn_145817960CustomGradient-145816104B
&__inference_internal_grad_fn_145817978CustomGradient-145816128B
&__inference_internal_grad_fn_145817996CustomGradient-145816153B
&__inference_internal_grad_fn_145818014CustomGradient-145816177B
&__inference_internal_grad_fn_145818032CustomGradient-145816201B
&__inference_internal_grad_fn_145818050CustomGradient-145816226B
&__inference_internal_grad_fn_145818068CustomGradient-145816250B
&__inference_internal_grad_fn_145818086CustomGradient-145816299B
&__inference_internal_grad_fn_145818104CustomGradient-145816323B
&__inference_internal_grad_fn_145818122CustomGradient-145816347B
&__inference_internal_grad_fn_145818140CustomGradient-145817010B
&__inference_internal_grad_fn_145818158CustomGradient-145817024B
&__inference_internal_grad_fn_145818176CustomGradient-145817039B
&__inference_internal_grad_fn_145818194CustomGradient-145817053B
&__inference_internal_grad_fn_145818212CustomGradient-145817067B
&__inference_internal_grad_fn_145818230CustomGradient-145817082B
&__inference_internal_grad_fn_145818248CustomGradient-145817096B
&__inference_internal_grad_fn_145818266CustomGradient-145817116B
&__inference_internal_grad_fn_145818284CustomGradient-145817130B
&__inference_internal_grad_fn_145818302CustomGradient-145817144B
&__inference_internal_grad_fn_145818320CustomGradient-145817169B
&__inference_internal_grad_fn_145818338CustomGradient-145817183B
&__inference_internal_grad_fn_145818356CustomGradient-145817198B
&__inference_internal_grad_fn_145818374CustomGradient-145817212B
&__inference_internal_grad_fn_145818392CustomGradient-145817226B
&__inference_internal_grad_fn_145818410CustomGradient-145817241B
&__inference_internal_grad_fn_145818428CustomGradient-145817255B
&__inference_internal_grad_fn_145818446CustomGradient-145817275B
&__inference_internal_grad_fn_145818464CustomGradient-145817289B
&__inference_internal_grad_fn_145818482CustomGradient-145817303B
&__inference_internal_grad_fn_145818500CustomGradient-145817388B
&__inference_internal_grad_fn_145818518CustomGradient-145817415B
&__inference_internal_grad_fn_145818536CustomGradient-145817452B
&__inference_internal_grad_fn_145818554CustomGradient-145817479B
&__inference_internal_grad_fn_145818572CustomGradient-145817516B
&__inference_internal_grad_fn_145818590CustomGradient-145817543B
&__inference_internal_grad_fn_145818608CustomGradient-145817570B
&__inference_internal_grad_fn_145818626CustomGradient-145817632B
&__inference_internal_grad_fn_145818644CustomGradient-145817659B
&__inference_internal_grad_fn_145818662CustomGradient-145817686"ÛL
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Á
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
ú2÷
+__inference_model_2_layer_call_fn_145816425
+__inference_model_2_layer_call_fn_145816949
+__inference_model_2_layer_call_fn_145816999
+__inference_model_2_layer_call_fn_145816769À
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
æ2ã
F__inference_model_2_layer_call_and_return_conditional_losses_145817158
F__inference_model_2_layer_call_and_return_conditional_losses_145817317
F__inference_model_2_layer_call_and_return_conditional_losses_145816834
F__inference_model_2_layer_call_and_return_conditional_losses_145816899À
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
ØBÕ
$__inference__wrapped_model_145816061input_5input_6"
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
×2Ô
-__inference_conv2d_10_layer_call_fn_145817378¢
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
ò2ï
H__inference_conv2d_10_layer_call_and_return_conditional_losses_145817396¢
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
×2Ô
-__inference_conv2d_11_layer_call_fn_145817405¢
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
ò2ï
H__inference_conv2d_11_layer_call_and_return_conditional_losses_145817423¢
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
Ý2Ú
3__inference_max_pooling2d_4_layer_call_fn_145817428¢
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
ø2õ
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145817433¢
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
×2Ô
-__inference_conv2d_12_layer_call_fn_145817442¢
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
ò2ï
H__inference_conv2d_12_layer_call_and_return_conditional_losses_145817460¢
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
×2Ô
-__inference_conv2d_13_layer_call_fn_145817469¢
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
ò2ï
H__inference_conv2d_13_layer_call_and_return_conditional_losses_145817487¢
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
Ý2Ú
3__inference_max_pooling2d_5_layer_call_fn_145817492¢
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
ø2õ
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145817497¢
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
Ö2Ó
,__inference_dense_12_layer_call_fn_145817506¢
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
G__inference_dense_12_layer_call_and_return_conditional_losses_145817524¢
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
×2Ô
-__inference_conv2d_14_layer_call_fn_145817533¢
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
ò2ï
H__inference_conv2d_14_layer_call_and_return_conditional_losses_145817551¢
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
Ö2Ó
,__inference_dense_13_layer_call_fn_145817560¢
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
G__inference_dense_13_layer_call_and_return_conditional_losses_145817578¢
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
×2Ô
-__inference_flatten_4_layer_call_fn_145817583¢
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
ò2ï
H__inference_flatten_4_layer_call_and_return_conditional_losses_145817589¢
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
×2Ô
-__inference_flatten_5_layer_call_fn_145817594¢
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
ò2ï
H__inference_flatten_5_layer_call_and_return_conditional_losses_145817600¢
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
Û2Ø
1__inference_concatenate_2_layer_call_fn_145817606¢
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
ö2ó
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145817613¢
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
Ö2Ó
,__inference_dense_14_layer_call_fn_145817622¢
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
G__inference_dense_14_layer_call_and_return_conditional_losses_145817640¢
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
Ö2Ó
,__inference_dense_15_layer_call_fn_145817649¢
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
G__inference_dense_15_layer_call_and_return_conditional_losses_145817667¢
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
Ö2Ó
,__inference_dense_16_layer_call_fn_145817676¢
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
G__inference_dense_16_layer_call_and_return_conditional_losses_145817694¢
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
Ö2Ó
,__inference_dense_17_layer_call_fn_145817703¢
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
G__inference_dense_17_layer_call_and_return_conditional_losses_145817713¢
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
ÕBÒ
'__inference_signature_wrapper_145817369input_5input_6"
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
Bb@
model_2/conv2d_10/beta:0$__inference__wrapped_model_145816061
EbC
model_2/conv2d_10/BiasAdd:0$__inference__wrapped_model_145816061
Bb@
model_2/conv2d_11/beta:0$__inference__wrapped_model_145816061
EbC
model_2/conv2d_11/BiasAdd:0$__inference__wrapped_model_145816061
Bb@
model_2/conv2d_12/beta:0$__inference__wrapped_model_145816061
EbC
model_2/conv2d_12/BiasAdd:0$__inference__wrapped_model_145816061
Bb@
model_2/conv2d_13/beta:0$__inference__wrapped_model_145816061
EbC
model_2/conv2d_13/BiasAdd:0$__inference__wrapped_model_145816061
Ab?
model_2/dense_12/beta:0$__inference__wrapped_model_145816061
DbB
model_2/dense_12/BiasAdd:0$__inference__wrapped_model_145816061
Ab?
model_2/dense_13/beta:0$__inference__wrapped_model_145816061
DbB
model_2/dense_13/BiasAdd:0$__inference__wrapped_model_145816061
Bb@
model_2/conv2d_14/beta:0$__inference__wrapped_model_145816061
EbC
model_2/conv2d_14/BiasAdd:0$__inference__wrapped_model_145816061
Ab?
model_2/dense_14/beta:0$__inference__wrapped_model_145816061
DbB
model_2/dense_14/BiasAdd:0$__inference__wrapped_model_145816061
Ab?
model_2/dense_15/beta:0$__inference__wrapped_model_145816061
DbB
model_2/dense_15/BiasAdd:0$__inference__wrapped_model_145816061
Ab?
model_2/dense_16/beta:0$__inference__wrapped_model_145816061
DbB
model_2/dense_16/BiasAdd:0$__inference__wrapped_model_145816061
TbR
beta:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112
WbU
	BiasAdd:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_145816112
TbR
beta:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136
WbU
	BiasAdd:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_145816136
TbR
beta:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161
WbU
	BiasAdd:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_145816161
TbR
beta:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185
WbU
	BiasAdd:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_145816185
SbQ
beta:0G__inference_dense_12_layer_call_and_return_conditional_losses_145816209
VbT
	BiasAdd:0G__inference_dense_12_layer_call_and_return_conditional_losses_145816209
SbQ
beta:0G__inference_dense_13_layer_call_and_return_conditional_losses_145816234
VbT
	BiasAdd:0G__inference_dense_13_layer_call_and_return_conditional_losses_145816234
TbR
beta:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258
WbU
	BiasAdd:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_145816258
SbQ
beta:0G__inference_dense_14_layer_call_and_return_conditional_losses_145816307
VbT
	BiasAdd:0G__inference_dense_14_layer_call_and_return_conditional_losses_145816307
SbQ
beta:0G__inference_dense_15_layer_call_and_return_conditional_losses_145816331
VbT
	BiasAdd:0G__inference_dense_15_layer_call_and_return_conditional_losses_145816331
SbQ
beta:0G__inference_dense_16_layer_call_and_return_conditional_losses_145816355
VbT
	BiasAdd:0G__inference_dense_16_layer_call_and_return_conditional_losses_145816355
\bZ
conv2d_10/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
_b]
conv2d_10/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
\bZ
conv2d_11/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
_b]
conv2d_11/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
\bZ
conv2d_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
_b]
conv2d_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
\bZ
conv2d_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
_b]
conv2d_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
[bY
dense_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
^b\
dense_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
[bY
dense_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
^b\
dense_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
\bZ
conv2d_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
_b]
conv2d_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
[bY
dense_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
^b\
dense_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
[bY
dense_15/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
^b\
dense_15/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
[bY
dense_16/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
^b\
dense_16/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817158
\bZ
conv2d_10/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
_b]
conv2d_10/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
\bZ
conv2d_11/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
_b]
conv2d_11/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
\bZ
conv2d_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
_b]
conv2d_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
\bZ
conv2d_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
_b]
conv2d_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
[bY
dense_12/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
^b\
dense_12/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
[bY
dense_13/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
^b\
dense_13/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
\bZ
conv2d_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
_b]
conv2d_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
[bY
dense_14/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
^b\
dense_14/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
[bY
dense_15/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
^b\
dense_15/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
[bY
dense_16/beta:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
^b\
dense_16/BiasAdd:0F__inference_model_2_layer_call_and_return_conditional_losses_145817317
TbR
beta:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_145817396
WbU
	BiasAdd:0H__inference_conv2d_10_layer_call_and_return_conditional_losses_145817396
TbR
beta:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_145817423
WbU
	BiasAdd:0H__inference_conv2d_11_layer_call_and_return_conditional_losses_145817423
TbR
beta:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_145817460
WbU
	BiasAdd:0H__inference_conv2d_12_layer_call_and_return_conditional_losses_145817460
TbR
beta:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_145817487
WbU
	BiasAdd:0H__inference_conv2d_13_layer_call_and_return_conditional_losses_145817487
SbQ
beta:0G__inference_dense_12_layer_call_and_return_conditional_losses_145817524
VbT
	BiasAdd:0G__inference_dense_12_layer_call_and_return_conditional_losses_145817524
TbR
beta:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_145817551
WbU
	BiasAdd:0H__inference_conv2d_14_layer_call_and_return_conditional_losses_145817551
SbQ
beta:0G__inference_dense_13_layer_call_and_return_conditional_losses_145817578
VbT
	BiasAdd:0G__inference_dense_13_layer_call_and_return_conditional_losses_145817578
SbQ
beta:0G__inference_dense_14_layer_call_and_return_conditional_losses_145817640
VbT
	BiasAdd:0G__inference_dense_14_layer_call_and_return_conditional_losses_145817640
SbQ
beta:0G__inference_dense_15_layer_call_and_return_conditional_losses_145817667
VbT
	BiasAdd:0G__inference_dense_15_layer_call_and_return_conditional_losses_145817667
SbQ
beta:0G__inference_dense_16_layer_call_and_return_conditional_losses_145817694
VbT
	BiasAdd:0G__inference_dense_16_layer_call_and_return_conditional_losses_145817694à
$__inference__wrapped_model_145816061·&'67?@PQbcYZ`¢]
V¢S
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿÖ
L__inference_concatenate_2_layer_call_and_return_conditional_losses_145817613[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 ­
1__inference_concatenate_2_layer_call_fn_145817606x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¹
H__inference_conv2d_10_layer_call_and_return_conditional_losses_145817396m7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 
-__inference_conv2d_10_layer_call_fn_145817378`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª "!ÿÿÿÿÿÿÿÿÿ		º
H__inference_conv2d_11_layer_call_and_return_conditional_losses_145817423n&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ		
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 
-__inference_conv2d_11_layer_call_fn_145817405a&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ		
ª "!ÿÿÿÿÿÿÿÿÿ		º
H__inference_conv2d_12_layer_call_and_return_conditional_losses_145817460n678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_conv2d_12_layer_call_fn_145817442a678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¹
H__inference_conv2d_13_layer_call_and_return_conditional_losses_145817487m?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_conv2d_13_layer_call_fn_145817469`?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@¸
H__inference_conv2d_14_layer_call_and_return_conditional_losses_145817551lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_conv2d_14_layer_call_fn_145817533_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@§
G__inference_dense_12_layer_call_and_return_conditional_losses_145817524\PQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_dense_12_layer_call_fn_145817506OPQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@§
G__inference_dense_13_layer_call_and_return_conditional_losses_145817578\bc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_dense_13_layer_call_fn_145817560Obc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ «
G__inference_dense_14_layer_call_and_return_conditional_losses_145817640`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_14_layer_call_fn_145817622S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ«
G__inference_dense_15_layer_call_and_return_conditional_losses_145817667`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_15_layer_call_fn_145817649S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
G__inference_dense_16_layer_call_and_return_conditional_losses_145817694_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_dense_16_layer_call_fn_145817676R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@©
G__inference_dense_17_layer_call_and_return_conditional_losses_145817713^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_17_layer_call_fn_145817703Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ­
H__inference_flatten_4_layer_call_and_return_conditional_losses_145817589a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_flatten_4_layer_call_fn_145817583T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¤
H__inference_flatten_5_layer_call_and_return_conditional_losses_145817600X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 |
-__inference_flatten_5_layer_call_fn_145817594K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ Ù
&__inference_internal_grad_fn_145817780®úûw¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145817798®üýw¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145817816®þÿw¢t
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
1ÿÿÿÿÿÿÿÿÿÖ
&__inference_internal_grad_fn_145817834«u¢r
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145817852e¢b
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145817870e¢b
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
1ÿÿÿÿÿÿÿÿÿ Ö
&__inference_internal_grad_fn_145817888«u¢r
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
1ÿÿÿÿÿÿÿÿÿ@Á
&__inference_internal_grad_fn_145817906g¢d
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
1ÿÿÿÿÿÿÿÿÿÁ
&__inference_internal_grad_fn_145817924g¢d
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
1ÿÿÿÿÿÿÿÿÿ¾
&__inference_internal_grad_fn_145817942e¢b
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
1ÿÿÿÿÿÿÿÿÿ@Ù
&__inference_internal_grad_fn_145817960®w¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145817978®w¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145817996®w¢t
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
1ÿÿÿÿÿÿÿÿÿÖ
&__inference_internal_grad_fn_145818014«u¢r
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818032e¢b
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818050e¢b
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
1ÿÿÿÿÿÿÿÿÿ Ö
&__inference_internal_grad_fn_145818068«u¢r
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
1ÿÿÿÿÿÿÿÿÿ@Á
&__inference_internal_grad_fn_145818086g¢d
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
1ÿÿÿÿÿÿÿÿÿÁ
&__inference_internal_grad_fn_145818104g¢d
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
1ÿÿÿÿÿÿÿÿÿ¾
&__inference_internal_grad_fn_145818122 ¡e¢b
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
1ÿÿÿÿÿÿÿÿÿ@Ù
&__inference_internal_grad_fn_145818140®¢£w¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145818158®¤¥w¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145818176®¦§w¢t
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
1ÿÿÿÿÿÿÿÿÿÖ
&__inference_internal_grad_fn_145818194«¨©u¢r
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818212ª«e¢b
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818230¬­e¢b
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
1ÿÿÿÿÿÿÿÿÿ Ö
&__inference_internal_grad_fn_145818248«®¯u¢r
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
1ÿÿÿÿÿÿÿÿÿ@Á
&__inference_internal_grad_fn_145818266°±g¢d
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
1ÿÿÿÿÿÿÿÿÿÁ
&__inference_internal_grad_fn_145818284²³g¢d
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
1ÿÿÿÿÿÿÿÿÿ¾
&__inference_internal_grad_fn_145818302´µe¢b
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
1ÿÿÿÿÿÿÿÿÿ@Ù
&__inference_internal_grad_fn_145818320®¶·w¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145818338®¸¹w¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145818356®º»w¢t
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
1ÿÿÿÿÿÿÿÿÿÖ
&__inference_internal_grad_fn_145818374«¼½u¢r
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818392¾¿e¢b
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818410ÀÁe¢b
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
1ÿÿÿÿÿÿÿÿÿ Ö
&__inference_internal_grad_fn_145818428«ÂÃu¢r
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
1ÿÿÿÿÿÿÿÿÿ@Á
&__inference_internal_grad_fn_145818446ÄÅg¢d
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
1ÿÿÿÿÿÿÿÿÿÁ
&__inference_internal_grad_fn_145818464ÆÇg¢d
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
1ÿÿÿÿÿÿÿÿÿ¾
&__inference_internal_grad_fn_145818482ÈÉe¢b
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
1ÿÿÿÿÿÿÿÿÿ@Ù
&__inference_internal_grad_fn_145818500®ÊËw¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145818518®ÌÍw¢t
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
1ÿÿÿÿÿÿÿÿÿ		Ù
&__inference_internal_grad_fn_145818536®ÎÏw¢t
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
1ÿÿÿÿÿÿÿÿÿÖ
&__inference_internal_grad_fn_145818554«ÐÑu¢r
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818572ÒÓe¢b
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
1ÿÿÿÿÿÿÿÿÿ@Ö
&__inference_internal_grad_fn_145818590«ÔÕu¢r
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
1ÿÿÿÿÿÿÿÿÿ@¾
&__inference_internal_grad_fn_145818608Ö×e¢b
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
1ÿÿÿÿÿÿÿÿÿ Á
&__inference_internal_grad_fn_145818626ØÙg¢d
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
1ÿÿÿÿÿÿÿÿÿÁ
&__inference_internal_grad_fn_145818644ÚÛg¢d
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
1ÿÿÿÿÿÿÿÿÿ¾
&__inference_internal_grad_fn_145818662ÜÝe¢b
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
1ÿÿÿÿÿÿÿÿÿ@ñ
N__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_145817433R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_4_layer_call_fn_145817428R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿñ
N__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_145817497R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_5_layer_call_fn_145817492R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿü
F__inference_model_2_layer_call_and_return_conditional_losses_145816834±&'67?@PQbcYZh¢e
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
 ü
F__inference_model_2_layer_call_and_return_conditional_losses_145816899±&'67?@PQbcYZh¢e
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
 þ
F__inference_model_2_layer_call_and_return_conditional_losses_145817158³&'67?@PQbcYZj¢g
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
 þ
F__inference_model_2_layer_call_and_return_conditional_losses_145817317³&'67?@PQbcYZj¢g
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
 Ô
+__inference_model_2_layer_call_fn_145816425¤&'67?@PQbcYZh¢e
^¢[
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÔ
+__inference_model_2_layer_call_fn_145816769¤&'67?@PQbcYZh¢e
^¢[
QN
)&
input_5ÿÿÿÿÿÿÿÿÿ		
!
input_6ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÖ
+__inference_model_2_layer_call_fn_145816949¦&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÖ
+__inference_model_2_layer_call_fn_145816999¦&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿô
'__inference_signature_wrapper_145817369È&'67?@PQbcYZq¢n
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