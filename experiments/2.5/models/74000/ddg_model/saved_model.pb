î
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68äÚ

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@*
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

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
 *
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
 *
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	@*
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
¬c
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*çb
valueÝbBÚb BÓb
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
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
_Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ		
z
serving_default_input_2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ë
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasconv2d_4/kernelconv2d_4/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*#
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
&__inference_signature_wrapper_94896345
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¼
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
GPU2*0,1J 8 **
f%R#
!__inference__traced_save_94897679
ÿ
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
GPU2*0,1J 8 *-
f(R&
$__inference__traced_restore_94897755¹
ò

%__inference_internal_grad_fn_94896810
result_grads_0
result_grads_1
mul_model_conv2d_3_beta
mul_model_conv2d_3_biasadd
identity
mulMulmul_model_conv2d_3_betamul_model_conv2d_3_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
mul_1Mulmul_model_conv2d_3_betamul_model_conv2d_3_biasadd*
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
Ï

%__inference_internal_grad_fn_94897458
result_grads_0
result_grads_1
mul_dense_4_beta
mul_dense_4_biasadd
identityt
mulMulmul_dense_4_betamul_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
mul_1Mulmul_dense_4_betamul_dense_4_biasadd*
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
·
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_94895254

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


%__inference_internal_grad_fn_94896792
result_grads_0
result_grads_1
mul_model_conv2d_2_beta
mul_model_conv2d_2_biasadd
identity
mulMulmul_model_conv2d_2_betamul_model_conv2d_2_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
mul_1Mulmul_model_conv2d_2_betamul_model_conv2d_2_biasadd*
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
­
|
%__inference_internal_grad_fn_94897620
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

|
%__inference_internal_grad_fn_94897530
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
ú
£
+__inference_conv2d_1_layer_call_fn_94896381

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112x
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
È

&__inference_signature_wrapper_94896345
input_1
input_2"
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
#__inference__wrapped_model_94895037o
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
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ò

%__inference_internal_grad_fn_94896864
result_grads_0
result_grads_1
mul_model_conv2d_4_beta
mul_model_conv2d_4_biasadd
identity
mulMulmul_model_conv2d_4_betamul_model_conv2d_4_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@{
mul_1Mulmul_model_conv2d_4_betamul_model_conv2d_4_biasadd*
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
%__inference_internal_grad_fn_94897548
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
¥
|
%__inference_internal_grad_fn_94896954
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
åW
³
$__inference__traced_restore_94897755
file_prefix9
assignvariableop_conv2d_kernel:-
assignvariableop_1_conv2d_bias:	>
"assignvariableop_2_conv2d_1_kernel:/
 assignvariableop_3_conv2d_1_bias:	>
"assignvariableop_4_conv2d_2_kernel:/
 assignvariableop_5_conv2d_2_bias:	=
"assignvariableop_6_conv2d_3_kernel:@.
 assignvariableop_7_conv2d_3_bias:@1
assignvariableop_8_dense_kernel:@+
assignvariableop_9_dense_bias:@=
#assignvariableop_10_conv2d_4_kernel:@@/
!assignvariableop_11_conv2d_4_bias:@4
"assignvariableop_12_dense_1_kernel:@ .
 assignvariableop_13_dense_1_bias: 6
"assignvariableop_14_dense_2_kernel:
 /
 assignvariableop_15_dense_2_bias:	6
"assignvariableop_16_dense_3_kernel:
/
 assignvariableop_17_dense_3_bias:	5
"assignvariableop_18_dense_4_kernel:	@.
 assignvariableop_19_dense_4_bias:@4
"assignvariableop_20_dense_5_kernel:@.
 assignvariableop_21_dense_5_bias:
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
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_5_biasIdentity_21:output:0"/device:CPU:0*
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
Î

F__inference_conv2d_1_layer_call_and_return_conditional_losses_94896399

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
_gradient_op_typeCustomGradient-94896391*L
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
ª
ö
C__inference_dense_layer_call_and_return_conditional_losses_94895185

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
_gradient_op_typeCustomGradient-94895177*:
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
Ü

%__inference_internal_grad_fn_94897314
result_grads_0
result_grads_1
mul_conv2d_1_beta
mul_conv2d_1_biasadd
identity
mulMulmul_conv2d_1_betamul_conv2d_1_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		p
mul_1Mulmul_conv2d_1_betamul_conv2d_1_biasadd*
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
|
%__inference_internal_grad_fn_94897044
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
ê

(__inference_model_layer_call_fn_94895745
input_1
input_2"
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
identity¢StatefulPartitionedCall÷
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
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_94895648o
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
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
É
Ô
C__inference_model_layer_call_and_return_conditional_losses_94896293
inputs_0
inputs_1@
%conv2d_conv2d_readvariableop_resource:5
&conv2d_biasadd_readvariableop_resource:	C
'conv2d_1_conv2d_readvariableop_resource:7
(conv2d_1_biasadd_readvariableop_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	B
'conv2d_3_conv2d_readvariableop_resource:@6
(conv2d_3_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@:
&dense_2_matmul_readvariableop_resource:
 6
'dense_2_biasadd_readvariableop_resource:	:
&dense_3_matmul_readvariableop_resource:
6
'dense_3_biasadd_readvariableop_resource:	9
&dense_4_matmul_readvariableop_resource:	@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ª
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		P
conv2d/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{

conv2d/mulMulconv2d/beta:output:0conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		d
conv2d/SigmoidSigmoidconv2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		{
conv2d/mul_1Mulconv2d/BiasAdd:output:0conv2d/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		h
conv2d/IdentityIdentityconv2d/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ó
conv2d/IdentityN	IdentityNconv2d/mul_1:z:0conv2d/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896145*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¿
conv2d_1/Conv2DConv2Dconv2d/IdentityN:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		R
conv2d_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_1/mulMulconv2d_1/beta:output:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		h
conv2d_1/SigmoidSigmoidconv2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
conv2d_1/mul_1Mulconv2d_1/BiasAdd:output:0conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		l
conv2d_1/IdentityIdentityconv2d_1/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ù
conv2d_1/IdentityN	IdentityNconv2d_1/mul_1:z:0conv2d_1/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896159*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		«
max_pooling2d/MaxPoolMaxPoolconv2d_1/IdentityN:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
conv2d_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_2/mulMulconv2d_2/beta:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
conv2d_2/SigmoidSigmoidconv2d_2/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_2/mul_1Mulconv2d_2/BiasAdd:output:0conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_2/IdentityIdentityconv2d_2/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
conv2d_2/IdentityN	IdentityNconv2d_2/mul_1:z:0conv2d_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896174*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0À
conv2d_3/Conv2DConv2Dconv2d_2/IdentityN:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
conv2d_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_3/mulMulconv2d_3/beta:output:0conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
conv2d_3/SigmoidSigmoidconv2d_3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_3/mul_1Mulconv2d_3/BiasAdd:output:0conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
conv2d_3/IdentityIdentityconv2d_3/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
conv2d_3/IdentityN	IdentityNconv2d_3/mul_1:z:0conv2d_3/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896188*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0w
dense/MatMulMatMulinputs_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O

dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?o
	dense/mulMuldense/beta:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
dense/SigmoidSigmoiddense/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dense/mul_1Muldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
dense/IdentityIdentitydense/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
dense/IdentityN	IdentityNdense/mul_1:z:0dense/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896202*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@¬
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/IdentityN:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896217*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ 
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Å
conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
conv2d_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_4/mulMulconv2d_4/beta:output:0conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
conv2d_4/SigmoidSigmoidconv2d_4/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_4/mul_1Mulconv2d_4/BiasAdd:output:0conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
conv2d_4/IdentityIdentityconv2d_4/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
conv2d_4/IdentityN	IdentityNconv2d_4/mul_1:z:0conv2d_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896231*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshapeconv2d_4/IdentityN:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_1/ReshapeReshapedense_1/IdentityN:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_2/MatMulMatMulconcatenate/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?v
dense_2/mulMuldense_2/beta:output:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dense_2/SigmoidSigmoiddense_2/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
dense_2/mul_1Muldense_2/BiasAdd:output:0dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_2/IdentityIdentitydense_2/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
dense_2/IdentityN	IdentityNdense_2/mul_1:z:0dense_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896251*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_3/MatMulMatMuldense_2/IdentityN:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?v
dense_3/mulMuldense_3/beta:output:0dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dense_3/SigmoidSigmoiddense_3/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
dense_3/mul_1Muldense_3/BiasAdd:output:0dense_3/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_3/IdentityIdentitydense_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
dense_3/IdentityN	IdentityNdense_3/mul_1:z:0dense_3/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896265*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_4/MatMulMatMuldense_3/IdentityN:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
dense_4/mulMuldense_4/beta:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
dense_4/SigmoidSigmoiddense_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
dense_4/mul_1Muldense_4/BiasAdd:output:0dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dense_4/IdentityIdentitydense_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
dense_4/IdentityN	IdentityNdense_4/mul_1:z:0dense_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896279*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_5/MatMulMatMuldense_4/IdentityN:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2>
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
ó
 
+__inference_conv2d_4_layer_call_fn_94896509

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallè
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234w
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
Î

F__inference_conv2d_2_layer_call_and_return_conditional_losses_94896436

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
_gradient_op_typeCustomGradient-94896428*L
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
Ð

*__inference_dense_2_layer_call_fn_94896598

inputs
unknown:
 
	unknown_0:	
identity¢StatefulPartitionedCallà
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_94895283p
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
ûH


C__inference_model_layer_call_and_return_conditional_losses_94895810
input_1
input_2*
conv2d_94895749:
conv2d_94895751:	-
conv2d_1_94895754: 
conv2d_1_94895756:	-
conv2d_2_94895760: 
conv2d_2_94895762:	,
conv2d_3_94895765:@
conv2d_3_94895767:@ 
dense_94895770:@
dense_94895772:@"
dense_1_94895776:@ 
dense_1_94895778: +
conv2d_4_94895781:@@
conv2d_4_94895783:@$
dense_2_94895789:
 
dense_2_94895791:	$
dense_3_94895794:

dense_3_94895796:	#
dense_4_94895799:	@
dense_4_94895801:@"
dense_5_94895804:@
dense_5_94895806:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallý
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_94895749conv2d_94895751*
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
GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_94895088¥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_94895754conv2d_1_94895756*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112õ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94895046¤
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_94895760conv2d_2_94895762*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_94895765conv2d_3_94895767*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161ð
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_94895770dense_94895772*
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
GPU2*0,1J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_94895185ø
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94895058
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_94895776dense_1_94895778*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_94895210¥
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_94895781conv2d_4_94895783*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234á
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_94895246ã
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_94895254
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
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
GPU2*0,1J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_94895263
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_94895789dense_2_94895791*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_94895283
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_94895794dense_3_94895796*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_94895307
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_94895799dense_4_94895801*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_94895331
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_94895804dense_5_94895806*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_94895347w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
¶
F
*__inference_flatten_layer_call_fn_94896559

inputs
identity¶
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
GPU2*0,1J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_94895246a
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
ùH


C__inference_model_layer_call_and_return_conditional_losses_94895354

inputs
inputs_1*
conv2d_94895089:
conv2d_94895091:	-
conv2d_1_94895113: 
conv2d_1_94895115:	-
conv2d_2_94895138: 
conv2d_2_94895140:	,
conv2d_3_94895162:@
conv2d_3_94895164:@ 
dense_94895186:@
dense_94895188:@"
dense_1_94895211:@ 
dense_1_94895213: +
conv2d_4_94895235:@@
conv2d_4_94895237:@$
dense_2_94895284:
 
dense_2_94895286:	$
dense_3_94895308:

dense_3_94895310:	#
dense_4_94895332:	@
dense_4_94895334:@"
dense_5_94895348:@
dense_5_94895350:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallü
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_94895089conv2d_94895091*
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
GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_94895088¥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_94895113conv2d_1_94895115*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112õ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94895046¤
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_94895138conv2d_2_94895140*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_94895162conv2d_3_94895164*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161ñ
dense/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_94895186dense_94895188*
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
GPU2*0,1J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_94895185ø
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94895058
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_94895211dense_1_94895213*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_94895210¥
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_94895235conv2d_4_94895237*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234á
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_94895246ã
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_94895254
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
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
GPU2*0,1J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_94895263
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_94895284dense_2_94895286*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_94895283
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_94895308dense_3_94895310*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_94895307
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_94895332dense_4_94895334*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_94895331
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_94895348dense_5_94895350*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_94895347w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ

%__inference_internal_grad_fn_94897422
result_grads_0
result_grads_1
mul_dense_2_beta
mul_dense_2_biasadd
identityu
mulMulmul_dense_2_betamul_dense_2_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
mul_1Mulmul_dense_2_betamul_dense_2_biasadd*
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
Î

F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137

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
_gradient_op_typeCustomGradient-94895129*L
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
ð

(__inference_model_layer_call_fn_94895925
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
identity¢StatefulPartitionedCallù
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
GPU2*0,1J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_94895354o
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


%__inference_internal_grad_fn_94896774
result_grads_0
result_grads_1
mul_model_conv2d_1_beta
mul_model_conv2d_1_biasadd
identity
mulMulmul_model_conv2d_1_betamul_model_conv2d_1_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		|
mul_1Mulmul_model_conv2d_1_betamul_model_conv2d_1_biasadd*
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


%__inference_internal_grad_fn_94896882
result_grads_0
result_grads_1
mul_model_dense_2_beta
mul_model_dense_2_biasadd
identity
mulMulmul_model_dense_2_betamul_model_dense_2_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
mul_1Mulmul_model_dense_2_betamul_model_dense_2_biasadd*
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
¥
|
%__inference_internal_grad_fn_94896972
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
Þ

%__inference_internal_grad_fn_94897242
result_grads_0
result_grads_1
mul_dense_2_beta
mul_dense_2_biasadd
identityu
mulMulmul_dense_2_betamul_dense_2_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
mul_1Mulmul_dense_2_betamul_dense_2_biasadd*
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
Ã

%__inference_internal_grad_fn_94897368
result_grads_0
result_grads_1
mul_dense_beta
mul_dense_biasadd
identityp
mulMulmul_dense_betamul_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_1Mulmul_dense_betamul_dense_biasadd*
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
Ð

*__inference_dense_3_layer_call_fn_94896625

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallà
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_94895307p
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
ç

%__inference_internal_grad_fn_94896828
result_grads_0
result_grads_1
mul_model_dense_beta
mul_model_dense_biasadd
identity|
mulMulmul_model_dense_betamul_model_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
mul_1Mulmul_model_dense_betamul_model_dense_biasadd*
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


%__inference_internal_grad_fn_94896900
result_grads_0
result_grads_1
mul_model_dense_3_beta
mul_model_dense_3_biasadd
identity
mulMulmul_model_dense_3_betamul_model_dense_3_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
mul_1Mulmul_model_dense_3_betamul_model_dense_3_biasadd*
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
½
û
E__inference_dense_2_layer_call_and_return_conditional_losses_94895283

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
_gradient_op_typeCustomGradient-94895275*<
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
È

D__inference_conv2d_layer_call_and_return_conditional_losses_94895088

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
_gradient_op_typeCustomGradient-94895080*L
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
È	
ö
E__inference_dense_5_layer_call_and_return_conditional_losses_94896689

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
¥
|
%__inference_internal_grad_fn_94897494
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

|
%__inference_internal_grad_fn_94897008
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
¬
ø
E__inference_dense_1_layer_call_and_return_conditional_losses_94895210

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
_gradient_op_typeCustomGradient-94895202*:
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
°
ù
E__inference_dense_4_layer_call_and_return_conditional_losses_94895331

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
_gradient_op_typeCustomGradient-94895323*:
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

i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94896473

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
Ç
a
E__inference_flatten_layer_call_and_return_conditional_losses_94896565

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

g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94896409

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
Í

%__inference_internal_grad_fn_94897170
result_grads_0
result_grads_1
mul_conv2d_3_beta
mul_conv2d_3_biasadd
identity~
mulMulmul_conv2d_3_betamul_conv2d_3_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
mul_1Mulmul_conv2d_3_betamul_conv2d_3_biasadd*
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

(__inference_model_layer_call_fn_94895975
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
identity¢StatefulPartitionedCallù
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
GPU2*0,1J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_94895648o
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
È	
ö
E__inference_dense_5_layer_call_and_return_conditional_losses_94895347

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

g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94895046

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
%__inference_internal_grad_fn_94897512
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
Þ

%__inference_internal_grad_fn_94897440
result_grads_0
result_grads_1
mul_dense_3_beta
mul_dense_3_biasadd
identityu
mulMulmul_dense_3_betamul_dense_3_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
mul_1Mulmul_dense_3_betamul_dense_3_biasadd*
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
%__inference_internal_grad_fn_94897584
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
Ç
a
E__inference_flatten_layer_call_and_return_conditional_losses_94895246

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
%__inference_internal_grad_fn_94897062
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
Á

F__inference_conv2d_3_layer_call_and_return_conditional_losses_94896463

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
_gradient_op_typeCustomGradient-94896455*J
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
­
|
%__inference_internal_grad_fn_94897080
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
É
Ô
C__inference_model_layer_call_and_return_conditional_losses_94896134
inputs_0
inputs_1@
%conv2d_conv2d_readvariableop_resource:5
&conv2d_biasadd_readvariableop_resource:	C
'conv2d_1_conv2d_readvariableop_resource:7
(conv2d_1_biasadd_readvariableop_resource:	C
'conv2d_2_conv2d_readvariableop_resource:7
(conv2d_2_biasadd_readvariableop_resource:	B
'conv2d_3_conv2d_readvariableop_resource:@6
(conv2d_3_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@:
&dense_2_matmul_readvariableop_resource:
 6
'dense_2_biasadd_readvariableop_resource:	:
&dense_3_matmul_readvariableop_resource:
6
'dense_3_biasadd_readvariableop_resource:	9
&dense_4_matmul_readvariableop_resource:	@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ª
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		P
conv2d/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{

conv2d/mulMulconv2d/beta:output:0conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		d
conv2d/SigmoidSigmoidconv2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		{
conv2d/mul_1Mulconv2d/BiasAdd:output:0conv2d/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		h
conv2d/IdentityIdentityconv2d/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ó
conv2d/IdentityN	IdentityNconv2d/mul_1:z:0conv2d/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94895986*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0¿
conv2d_1/Conv2DConv2Dconv2d/IdentityN:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		R
conv2d_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_1/mulMulconv2d_1/beta:output:0conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		h
conv2d_1/SigmoidSigmoidconv2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
conv2d_1/mul_1Mulconv2d_1/BiasAdd:output:0conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		l
conv2d_1/IdentityIdentityconv2d_1/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		Ù
conv2d_1/IdentityN	IdentityNconv2d_1/mul_1:z:0conv2d_1/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896000*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		«
max_pooling2d/MaxPoolMaxPoolconv2d_1/IdentityN:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ä
conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
conv2d_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_2/mulMulconv2d_2/beta:output:0conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
conv2d_2/SigmoidSigmoidconv2d_2/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_2/mul_1Mulconv2d_2/BiasAdd:output:0conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_2/IdentityIdentityconv2d_2/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
conv2d_2/IdentityN	IdentityNconv2d_2/mul_1:z:0conv2d_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896015*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0À
conv2d_3/Conv2DConv2Dconv2d_2/IdentityN:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
conv2d_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_3/mulMulconv2d_3/beta:output:0conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
conv2d_3/SigmoidSigmoidconv2d_3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_3/mul_1Mulconv2d_3/BiasAdd:output:0conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
conv2d_3/IdentityIdentityconv2d_3/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
conv2d_3/IdentityN	IdentityNconv2d_3/mul_1:z:0conv2d_3/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896029*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0w
dense/MatMulMatMulinputs_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@O

dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?o
	dense/mulMuldense/beta:output:0dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
dense/SigmoidSigmoiddense/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dense/mul_1Muldense/BiasAdd:output:0dense/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
dense/IdentityIdentitydense/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¾
dense/IdentityN	IdentityNdense/mul_1:z:0dense/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896043*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@¬
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/IdentityN:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_1/MatMulMatMuldense/IdentityN:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Q
dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
dense_1/mulMuldense_1/beta:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ]
dense_1/SigmoidSigmoiddense_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ u
dense_1/mul_1Muldense_1/BiasAdd:output:0dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
dense_1/IdentityIdentitydense_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ä
dense_1/IdentityN	IdentityNdense_1/mul_1:z:0dense_1/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896058*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ 
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Å
conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@R
conv2d_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv2d_4/mulMulconv2d_4/beta:output:0conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
conv2d_4/SigmoidSigmoidconv2d_4/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_4/mul_1Mulconv2d_4/BiasAdd:output:0conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@k
conv2d_4/IdentityIdentityconv2d_4/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@×
conv2d_4/IdentityN	IdentityNconv2d_4/mul_1:z:0conv2d_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896072*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshapeconv2d_4/IdentityN:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_1/ReshapeReshapedense_1/IdentityN:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :²
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_2/MatMulMatMulconcatenate/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?v
dense_2/mulMuldense_2/beta:output:0dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dense_2/SigmoidSigmoiddense_2/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
dense_2/mul_1Muldense_2/BiasAdd:output:0dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_2/IdentityIdentitydense_2/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
dense_2/IdentityN	IdentityNdense_2/mul_1:z:0dense_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896092*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_3/MatMulMatMuldense_2/IdentityN:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?v
dense_3/mulMuldense_3/beta:output:0dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dense_3/SigmoidSigmoiddense_3/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
dense_3/mul_1Muldense_3/BiasAdd:output:0dense_3/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_3/IdentityIdentitydense_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
dense_3/IdentityN	IdentityNdense_3/mul_1:z:0dense_3/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896106*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_4/MatMulMatMuldense_3/IdentityN:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Q
dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?u
dense_4/mulMuldense_4/beta:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@]
dense_4/SigmoidSigmoiddense_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@u
dense_4/mul_1Muldense_4/BiasAdd:output:0dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
dense_4/IdentityIdentitydense_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
dense_4/IdentityN	IdentityNdense_4/mul_1:z:0dense_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94896120*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_5/MatMulMatMuldense_4/IdentityN:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2>
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
ô

%__inference_internal_grad_fn_94896918
result_grads_0
result_grads_1
mul_model_dense_4_beta
mul_model_dense_4_biasadd
identity
mulMulmul_model_dense_4_betamul_model_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
mul_1Mulmul_model_dense_4_betamul_model_dense_4_biasadd*
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
Æ
u
I__inference_concatenate_layer_call_and_return_conditional_losses_94896589
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
ú
£
+__inference_conv2d_2_layer_call_fn_94896418

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallé
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137x
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
Ï

%__inference_internal_grad_fn_94897278
result_grads_0
result_grads_1
mul_dense_4_beta
mul_dense_4_biasadd
identityt
mulMulmul_dense_4_betamul_dense_4_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
mul_1Mulmul_dense_4_betamul_dense_4_biasadd*
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

i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94895058

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
½

F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234

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
_gradient_op_typeCustomGradient-94895226*J
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
É

*__inference_dense_5_layer_call_fn_94896679

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallß
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_94895347o
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
Ð

%__inference_internal_grad_fn_94897296
result_grads_0
result_grads_1
mul_conv2d_beta
mul_conv2d_biasadd
identity{
mulMulmul_conv2d_betamul_conv2d_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		l
mul_1Mulmul_conv2d_betamul_conv2d_biasadd*
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
¨
H
,__inference_flatten_1_layer_call_fn_94896570

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
G__inference_flatten_1_layer_call_and_return_conditional_losses_94895254`
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
Ï

%__inference_internal_grad_fn_94897206
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
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
°
ù
E__inference_dense_4_layer_call_and_return_conditional_losses_94896670

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
_gradient_op_typeCustomGradient-94896662*:
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
¬
º
#__inference__wrapped_model_94895037
input_1
input_2F
+model_conv2d_conv2d_readvariableop_resource:;
,model_conv2d_biasadd_readvariableop_resource:	I
-model_conv2d_1_conv2d_readvariableop_resource:=
.model_conv2d_1_biasadd_readvariableop_resource:	I
-model_conv2d_2_conv2d_readvariableop_resource:=
.model_conv2d_2_biasadd_readvariableop_resource:	H
-model_conv2d_3_conv2d_readvariableop_resource:@<
.model_conv2d_3_biasadd_readvariableop_resource:@<
*model_dense_matmul_readvariableop_resource:@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@ ;
-model_dense_1_biasadd_readvariableop_resource: G
-model_conv2d_4_conv2d_readvariableop_resource:@@<
.model_conv2d_4_biasadd_readvariableop_resource:@@
,model_dense_2_matmul_readvariableop_resource:
 <
-model_dense_2_biasadd_readvariableop_resource:	@
,model_dense_3_matmul_readvariableop_resource:
<
-model_dense_3_biasadd_readvariableop_resource:	?
,model_dense_4_matmul_readvariableop_resource:	@;
-model_dense_4_biasadd_readvariableop_resource:@>
,model_dense_5_matmul_readvariableop_resource:@;
-model_dense_5_biasadd_readvariableop_resource:
identity¢#model/conv2d/BiasAdd/ReadVariableOp¢"model/conv2d/Conv2D/ReadVariableOp¢%model/conv2d_1/BiasAdd/ReadVariableOp¢$model/conv2d_1/Conv2D/ReadVariableOp¢%model/conv2d_2/BiasAdd/ReadVariableOp¢$model/conv2d_2/Conv2D/ReadVariableOp¢%model/conv2d_3/BiasAdd/ReadVariableOp¢$model/conv2d_3/Conv2D/ReadVariableOp¢%model/conv2d_4/BiasAdd/ReadVariableOp¢$model/conv2d_4/Conv2D/ReadVariableOp¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢#model/dense_1/MatMul/ReadVariableOp¢$model/dense_2/BiasAdd/ReadVariableOp¢#model/dense_2/MatMul/ReadVariableOp¢$model/dense_3/BiasAdd/ReadVariableOp¢#model/dense_3/MatMul/ReadVariableOp¢$model/dense_4/BiasAdd/ReadVariableOp¢#model/dense_4/MatMul/ReadVariableOp¢$model/dense_5/BiasAdd/ReadVariableOp¢#model/dense_5/MatMul/ReadVariableOp
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0µ
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
model/conv2d/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/conv2d/mulMulmodel/conv2d/beta:output:0model/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		p
model/conv2d/SigmoidSigmoidmodel/conv2d/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
model/conv2d/mul_1Mulmodel/conv2d/BiasAdd:output:0model/conv2d/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		t
model/conv2d/IdentityIdentitymodel/conv2d/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		å
model/conv2d/IdentityN	IdentityNmodel/conv2d/mul_1:z:0model/conv2d/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894889*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ñ
model/conv2d_1/Conv2DConv2Dmodel/conv2d/IdentityN:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
paddingSAME*
strides

%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0«
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		X
model/conv2d_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/conv2d_1/mulMulmodel/conv2d_1/beta:output:0model/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		t
model/conv2d_1/SigmoidSigmoidmodel/conv2d_1/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
model/conv2d_1/mul_1Mulmodel/conv2d_1/BiasAdd:output:0model/conv2d_1/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		x
model/conv2d_1/IdentityIdentitymodel/conv2d_1/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		ë
model/conv2d_1/IdentityN	IdentityNmodel/conv2d_1/mul_1:z:0model/conv2d_1/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894903*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ		·
model/max_pooling2d/MaxPoolMaxPool!model/conv2d_1/IdentityN:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ö
model/conv2d_2/Conv2DConv2D$model/max_pooling2d/MaxPool:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0«
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
model/conv2d_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/conv2d_2/mulMulmodel/conv2d_2/beta:output:0model/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
model/conv2d_2/SigmoidSigmoidmodel/conv2d_2/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/conv2d_2/mul_1Mulmodel/conv2d_2/BiasAdd:output:0model/conv2d_2/Sigmoid:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
model/conv2d_2/IdentityIdentitymodel/conv2d_2/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
model/conv2d_2/IdentityN	IdentityNmodel/conv2d_2/mul_1:z:0model/conv2d_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894918*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ò
model/conv2d_3/Conv2DConv2D!model/conv2d_2/IdentityN:output:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
model/conv2d_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/conv2d_3/mulMulmodel/conv2d_3/beta:output:0model/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
model/conv2d_3/SigmoidSigmoidmodel/conv2d_3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model/conv2d_3/mul_1Mulmodel/conv2d_3/BiasAdd:output:0model/conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
model/conv2d_3/IdentityIdentitymodel/conv2d_3/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@é
model/conv2d_3/IdentityN	IdentityNmodel/conv2d_3/mul_1:z:0model/conv2d_3/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894932*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
model/dense/MatMulMatMulinput_2)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
model/dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense/mulMulmodel/dense/beta:output:0model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
model/dense/SigmoidSigmoidmodel/dense/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model/dense/mul_1Mulmodel/dense/BiasAdd:output:0model/dense/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
model/dense/IdentityIdentitymodel/dense/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ð
model/dense/IdentityN	IdentityNmodel/dense/mul_1:z:0model/dense/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894946*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@¸
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_3/IdentityN:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
model/dense_1/MatMulMatMulmodel/dense/IdentityN:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0 
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ W
model/dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_1/mulMulmodel/dense_1/beta:output:0model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
model/dense_1/SigmoidSigmoidmodel/dense_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model/dense_1/mul_1Mulmodel/dense_1/BiasAdd:output:0model/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ m
model/dense_1/IdentityIdentitymodel/dense_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ö
model/dense_1/IdentityN	IdentityNmodel/dense_1/mul_1:z:0model/dense_1/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894961*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ 
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0×
model/conv2d_4/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
model/conv2d_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/conv2d_4/mulMulmodel/conv2d_4/beta:output:0model/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@s
model/conv2d_4/SigmoidSigmoidmodel/conv2d_4/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model/conv2d_4/mul_1Mulmodel/conv2d_4/BiasAdd:output:0model/conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
model/conv2d_4/IdentityIdentitymodel/conv2d_4/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@é
model/conv2d_4/IdentityN	IdentityNmodel/conv2d_4/mul_1:z:0model/conv2d_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894975*J
_output_shapes8
6:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
model/flatten/ReshapeReshape!model/conv2d_4/IdentityN:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
model/flatten_1/ReshapeReshape model/dense_1/IdentityN:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ê
model/concatenate/concatConcatV2model/flatten/Reshape:output:0 model/flatten_1/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0¡
model/dense_2/MatMulMatMul!model/concatenate/concat:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
model/dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_2/mulMulmodel/dense_2/beta:output:0model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
model/dense_2/SigmoidSigmoidmodel/dense_2/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dense_2/mul_1Mulmodel/dense_2/BiasAdd:output:0model/dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
model/dense_2/IdentityIdentitymodel/dense_2/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
model/dense_2/IdentityN	IdentityNmodel/dense_2/mul_1:z:0model/dense_2/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94894995*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
model/dense_3/MatMulMatMul model/dense_2/IdentityN:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
model/dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_3/mulMulmodel/dense_3/beta:output:0model/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
model/dense_3/SigmoidSigmoidmodel/dense_3/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dense_3/mul_1Mulmodel/dense_3/BiasAdd:output:0model/dense_3/Sigmoid:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
model/dense_3/IdentityIdentitymodel/dense_3/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
model/dense_3/IdentityN	IdentityNmodel/dense_3/mul_1:z:0model/dense_3/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94895009*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
model/dense_4/MatMulMatMul model/dense_3/IdentityN:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
model/dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_4/mulMulmodel/dense_4/beta:output:0model/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
model/dense_4/SigmoidSigmoidmodel/dense_4/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model/dense_4/mul_1Mulmodel/dense_4/BiasAdd:output:0model/dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@m
model/dense_4/IdentityIdentitymodel/dense_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ö
model/dense_4/IdentityN	IdentityNmodel/dense_4/mul_1:z:0model/dense_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-94895023*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
model/dense_5/MatMulMatMul model/dense_4/IdentityN:output:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentitymodel/dense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2J
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
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ô

%__inference_internal_grad_fn_94896846
result_grads_0
result_grads_1
mul_model_dense_1_beta
mul_model_dense_1_biasadd
identity
mulMulmul_model_dense_1_betamul_model_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
mul_1Mulmul_model_dense_1_betamul_model_dense_1_biasadd*
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
½
L
0__inference_max_pooling2d_layer_call_fn_94896404

inputs
identityÞ
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
GPU2*0,1J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94895046
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
Á

F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161

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
_gradient_op_typeCustomGradient-94895153*J
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
Ì

*__inference_dense_4_layer_call_fn_94896652

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallß
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_94895331o
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
¬
ø
E__inference_dense_1_layer_call_and_return_conditional_losses_94896554

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
_gradient_op_typeCustomGradient-94896546*:
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
Í

%__inference_internal_grad_fn_94897404
result_grads_0
result_grads_1
mul_conv2d_4_beta
mul_conv2d_4_biasadd
identity~
mulMulmul_conv2d_4_betamul_conv2d_4_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
mul_1Mulmul_conv2d_4_betamul_conv2d_4_biasadd*
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
â2
ä
!__inference__traced_save_94897679
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
value8B6B B B B B B B B B B B B B B B B B B B B B B B ä
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
Ï

%__inference_internal_grad_fn_94897386
result_grads_0
result_grads_1
mul_dense_1_beta
mul_dense_1_biasadd
identityt
mulMulmul_dense_1_betamul_dense_1_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ e
mul_1Mulmul_dense_1_betamul_dense_1_biasadd*
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
%__inference_internal_grad_fn_94897476
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
ª
ö
C__inference_dense_layer_call_and_return_conditional_losses_94896500

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
_gradient_op_typeCustomGradient-94896492*:
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

|
%__inference_internal_grad_fn_94896990
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
Å

(__inference_dense_layer_call_fn_94896482

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallÝ
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
GPU2*0,1J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_94895185o
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
ûH


C__inference_model_layer_call_and_return_conditional_losses_94895875
input_1
input_2*
conv2d_94895814:
conv2d_94895816:	-
conv2d_1_94895819: 
conv2d_1_94895821:	-
conv2d_2_94895825: 
conv2d_2_94895827:	,
conv2d_3_94895830:@
conv2d_3_94895832:@ 
dense_94895835:@
dense_94895837:@"
dense_1_94895841:@ 
dense_1_94895843: +
conv2d_4_94895846:@@
conv2d_4_94895848:@$
dense_2_94895854:
 
dense_2_94895856:	$
dense_3_94895859:

dense_3_94895861:	#
dense_4_94895864:	@
dense_4_94895866:@"
dense_5_94895869:@
dense_5_94895871:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallý
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_94895814conv2d_94895816*
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
GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_94895088¥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_94895819conv2d_1_94895821*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112õ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94895046¤
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_94895825conv2d_2_94895827*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_94895830conv2d_3_94895832*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161ð
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_94895835dense_94895837*
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
GPU2*0,1J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_94895185ø
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94895058
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_94895841dense_1_94895843*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_94895210¥
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_94895846conv2d_4_94895848*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234á
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_94895246ã
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_94895254
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
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
GPU2*0,1J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_94895263
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_94895854dense_2_94895856*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_94895283
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_94895859dense_3_94895861*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_94895307
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_94895864dense_4_94895866*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_94895331
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_94895869dense_5_94895871*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_94895347w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ		
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ê

(__inference_model_layer_call_fn_94895401
input_1
input_2"
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
identity¢StatefulPartitionedCall÷
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
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_94895354o
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
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

|
%__inference_internal_grad_fn_94897026
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
Í

%__inference_internal_grad_fn_94897350
result_grads_0
result_grads_1
mul_conv2d_3_beta
mul_conv2d_3_biasadd
identity~
mulMulmul_conv2d_3_betamul_conv2d_3_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
mul_1Mulmul_conv2d_3_betamul_conv2d_3_biasadd*
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
õ

%__inference_internal_grad_fn_94896756
result_grads_0
result_grads_1
mul_model_conv2d_beta
mul_model_conv2d_biasadd
identity
mulMulmul_model_conv2d_betamul_model_conv2d_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		x
mul_1Mulmul_model_conv2d_betamul_model_conv2d_biasadd*
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
|
%__inference_internal_grad_fn_94897566
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_94896576

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
Ð

%__inference_internal_grad_fn_94897116
result_grads_0
result_grads_1
mul_conv2d_beta
mul_conv2d_biasadd
identity{
mulMulmul_conv2d_betamul_conv2d_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		l
mul_1Mulmul_conv2d_betamul_conv2d_biasadd*
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
Ü

%__inference_internal_grad_fn_94897332
result_grads_0
result_grads_1
mul_conv2d_2_beta
mul_conv2d_2_biasadd
identity
mulMulmul_conv2d_2_betamul_conv2d_2_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
mul_1Mulmul_conv2d_2_betamul_conv2d_2_biasadd*
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

|
%__inference_internal_grad_fn_94897638
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
µ
Z
.__inference_concatenate_layer_call_fn_94896582
inputs_0
inputs_1
identityÇ
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
GPU2*0,1J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_94895263a
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
Ã

%__inference_internal_grad_fn_94897188
result_grads_0
result_grads_1
mul_dense_beta
mul_dense_biasadd
identityp
mulMulmul_dense_betamul_dense_biasadd^result_grads_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
mul_1Mulmul_dense_betamul_dense_biasadd*
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
Ü

%__inference_internal_grad_fn_94897152
result_grads_0
result_grads_1
mul_conv2d_2_beta
mul_conv2d_2_biasadd
identity
mulMulmul_conv2d_2_betamul_conv2d_2_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
mul_1Mulmul_conv2d_2_betamul_conv2d_2_biasadd*
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
Á
N
2__inference_max_pooling2d_1_layer_call_fn_94896468

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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94895058
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
Þ

%__inference_internal_grad_fn_94897260
result_grads_0
result_grads_1
mul_dense_3_beta
mul_dense_3_biasadd
identityu
mulMulmul_dense_3_betamul_dense_3_biasadd^result_grads_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
SigmoidSigmoidmul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
mul_1Mulmul_dense_3_betamul_dense_3_biasadd*
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
È

D__inference_conv2d_layer_call_and_return_conditional_losses_94896372

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
_gradient_op_typeCustomGradient-94896364*L
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
½

F__inference_conv2d_4_layer_call_and_return_conditional_losses_94896527

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
_gradient_op_typeCustomGradient-94896519*J
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
¥
|
%__inference_internal_grad_fn_94896936
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
ö
¡
+__inference_conv2d_3_layer_call_fn_94896445

inputs"
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallè
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161w
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
É

*__inference_dense_1_layer_call_fn_94896536

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallß
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_94895210o
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

|
%__inference_internal_grad_fn_94897098
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
Ü

%__inference_internal_grad_fn_94897134
result_grads_0
result_grads_1
mul_conv2d_1_beta
mul_conv2d_1_biasadd
identity
mulMulmul_conv2d_1_betamul_conv2d_1_biasadd^result_grads_0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		V
SigmoidSigmoidmul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		p
mul_1Mulmul_conv2d_1_betamul_conv2d_1_biasadd*
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
Í

%__inference_internal_grad_fn_94897224
result_grads_0
result_grads_1
mul_conv2d_4_beta
mul_conv2d_4_biasadd
identity~
mulMulmul_conv2d_4_betamul_conv2d_4_biasadd^result_grads_0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@U
SigmoidSigmoidmul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
mul_1Mulmul_conv2d_4_betamul_conv2d_4_biasadd*
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
Î

F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112

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
_gradient_op_typeCustomGradient-94895104*L
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
ó
 
)__inference_conv2d_layer_call_fn_94896354

inputs"
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallç
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
GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_94895088x
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
¾
s
I__inference_concatenate_layer_call_and_return_conditional_losses_94895263

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
ùH


C__inference_model_layer_call_and_return_conditional_losses_94895648

inputs
inputs_1*
conv2d_94895587:
conv2d_94895589:	-
conv2d_1_94895592: 
conv2d_1_94895594:	-
conv2d_2_94895598: 
conv2d_2_94895600:	,
conv2d_3_94895603:@
conv2d_3_94895605:@ 
dense_94895608:@
dense_94895610:@"
dense_1_94895614:@ 
dense_1_94895616: +
conv2d_4_94895619:@@
conv2d_4_94895621:@$
dense_2_94895627:
 
dense_2_94895629:	$
dense_3_94895632:

dense_3_94895634:	#
dense_4_94895637:	@
dense_4_94895639:@"
dense_5_94895642:@
dense_5_94895644:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallü
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_94895587conv2d_94895589*
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
GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_94895088¥
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_94895592conv2d_1_94895594*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112õ
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94895046¤
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_2_94895598conv2d_2_94895600*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_94895603conv2d_3_94895605*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161ñ
dense/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_94895608dense_94895610*
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
GPU2*0,1J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_94895185ø
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94895058
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_94895614dense_1_94895616*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_94895210¥
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_4_94895619conv2d_4_94895621*
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
GPU2*0,1J 8 *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234á
flatten/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_94895246ã
flatten_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_94895254
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
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
GPU2*0,1J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_94895263
dense_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_2_94895627dense_2_94895629*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_94895283
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_94895632dense_3_94895634*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_94895307
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_94895637dense_4_94895639*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_94895331
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_94895642dense_5_94895644*
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
GPU2*0,1J 8 *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_94895347w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:ÿÿÿÿÿÿÿÿÿ		:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
û
E__inference_dense_2_layer_call_and_return_conditional_losses_94896616

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
_gradient_op_typeCustomGradient-94896608*<
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
%__inference_internal_grad_fn_94897602
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
½
û
E__inference_dense_3_layer_call_and_return_conditional_losses_94896643

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
_gradient_op_typeCustomGradient-94896635*<
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
½
û
E__inference_dense_3_layer_call_and_return_conditional_losses_94895307

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
_gradient_op_typeCustomGradient-94895299*<
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
 
_user_specified_nameinputs@
%__inference_internal_grad_fn_94896756CustomGradient-94894889@
%__inference_internal_grad_fn_94896774CustomGradient-94894903@
%__inference_internal_grad_fn_94896792CustomGradient-94894918@
%__inference_internal_grad_fn_94896810CustomGradient-94894932@
%__inference_internal_grad_fn_94896828CustomGradient-94894946@
%__inference_internal_grad_fn_94896846CustomGradient-94894961@
%__inference_internal_grad_fn_94896864CustomGradient-94894975@
%__inference_internal_grad_fn_94896882CustomGradient-94894995@
%__inference_internal_grad_fn_94896900CustomGradient-94895009@
%__inference_internal_grad_fn_94896918CustomGradient-94895023@
%__inference_internal_grad_fn_94896936CustomGradient-94895080@
%__inference_internal_grad_fn_94896954CustomGradient-94895104@
%__inference_internal_grad_fn_94896972CustomGradient-94895129@
%__inference_internal_grad_fn_94896990CustomGradient-94895153@
%__inference_internal_grad_fn_94897008CustomGradient-94895177@
%__inference_internal_grad_fn_94897026CustomGradient-94895202@
%__inference_internal_grad_fn_94897044CustomGradient-94895226@
%__inference_internal_grad_fn_94897062CustomGradient-94895275@
%__inference_internal_grad_fn_94897080CustomGradient-94895299@
%__inference_internal_grad_fn_94897098CustomGradient-94895323@
%__inference_internal_grad_fn_94897116CustomGradient-94895986@
%__inference_internal_grad_fn_94897134CustomGradient-94896000@
%__inference_internal_grad_fn_94897152CustomGradient-94896015@
%__inference_internal_grad_fn_94897170CustomGradient-94896029@
%__inference_internal_grad_fn_94897188CustomGradient-94896043@
%__inference_internal_grad_fn_94897206CustomGradient-94896058@
%__inference_internal_grad_fn_94897224CustomGradient-94896072@
%__inference_internal_grad_fn_94897242CustomGradient-94896092@
%__inference_internal_grad_fn_94897260CustomGradient-94896106@
%__inference_internal_grad_fn_94897278CustomGradient-94896120@
%__inference_internal_grad_fn_94897296CustomGradient-94896145@
%__inference_internal_grad_fn_94897314CustomGradient-94896159@
%__inference_internal_grad_fn_94897332CustomGradient-94896174@
%__inference_internal_grad_fn_94897350CustomGradient-94896188@
%__inference_internal_grad_fn_94897368CustomGradient-94896202@
%__inference_internal_grad_fn_94897386CustomGradient-94896217@
%__inference_internal_grad_fn_94897404CustomGradient-94896231@
%__inference_internal_grad_fn_94897422CustomGradient-94896251@
%__inference_internal_grad_fn_94897440CustomGradient-94896265@
%__inference_internal_grad_fn_94897458CustomGradient-94896279@
%__inference_internal_grad_fn_94897476CustomGradient-94896364@
%__inference_internal_grad_fn_94897494CustomGradient-94896391@
%__inference_internal_grad_fn_94897512CustomGradient-94896428@
%__inference_internal_grad_fn_94897530CustomGradient-94896455@
%__inference_internal_grad_fn_94897548CustomGradient-94896492@
%__inference_internal_grad_fn_94897566CustomGradient-94896519@
%__inference_internal_grad_fn_94897584CustomGradient-94896546@
%__inference_internal_grad_fn_94897602CustomGradient-94896608@
%__inference_internal_grad_fn_94897620CustomGradient-94896635@
%__inference_internal_grad_fn_94897638CustomGradient-94896662"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ï
serving_defaultÛ
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ		
;
input_20
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ;
dense_50
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¼
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
î2ë
(__inference_model_layer_call_fn_94895401
(__inference_model_layer_call_fn_94895925
(__inference_model_layer_call_fn_94895975
(__inference_model_layer_call_fn_94895745À
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
Ú2×
C__inference_model_layer_call_and_return_conditional_losses_94896134
C__inference_model_layer_call_and_return_conditional_losses_94896293
C__inference_model_layer_call_and_return_conditional_losses_94895810
C__inference_model_layer_call_and_return_conditional_losses_94895875À
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
#__inference__wrapped_model_94895037input_1input_2"
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
(:&2conv2d/kernel
:2conv2d/bias
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
Ó2Ð
)__inference_conv2d_layer_call_fn_94896354¢
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
î2ë
D__inference_conv2d_layer_call_and_return_conditional_losses_94896372¢
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
+:)2conv2d_1/kernel
:2conv2d_1/bias
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
Õ2Ò
+__inference_conv2d_1_layer_call_fn_94896381¢
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_94896399¢
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
Ú2×
0__inference_max_pooling2d_layer_call_fn_94896404¢
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
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94896409¢
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
+:)2conv2d_2/kernel
:2conv2d_2/bias
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
Õ2Ò
+__inference_conv2d_2_layer_call_fn_94896418¢
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_94896436¢
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
*:(@2conv2d_3/kernel
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
Õ2Ò
+__inference_conv2d_3_layer_call_fn_94896445¢
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
F__inference_conv2d_3_layer_call_and_return_conditional_losses_94896463¢
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
2__inference_max_pooling2d_1_layer_call_fn_94896468¢
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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94896473¢
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
Ò2Ï
(__inference_dense_layer_call_fn_94896482¢
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
í2ê
C__inference_dense_layer_call_and_return_conditional_losses_94896500¢
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
Õ2Ò
+__inference_conv2d_4_layer_call_fn_94896509¢
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
F__inference_conv2d_4_layer_call_and_return_conditional_losses_94896527¢
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
Ô2Ñ
*__inference_dense_1_layer_call_fn_94896536¢
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
ï2ì
E__inference_dense_1_layer_call_and_return_conditional_losses_94896554¢
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
Ô2Ñ
*__inference_flatten_layer_call_fn_94896559¢
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
ï2ì
E__inference_flatten_layer_call_and_return_conditional_losses_94896565¢
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
,__inference_flatten_1_layer_call_fn_94896570¢
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
G__inference_flatten_1_layer_call_and_return_conditional_losses_94896576¢
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
Ø2Õ
.__inference_concatenate_layer_call_fn_94896582¢
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
ó2ð
I__inference_concatenate_layer_call_and_return_conditional_losses_94896589¢
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
": 
 2dense_2/kernel
:2dense_2/bias
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
Ô2Ñ
*__inference_dense_2_layer_call_fn_94896598¢
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
ï2ì
E__inference_dense_2_layer_call_and_return_conditional_losses_94896616¢
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
": 
2dense_3/kernel
:2dense_3/bias
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
Ô2Ñ
*__inference_dense_3_layer_call_fn_94896625¢
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
ï2ì
E__inference_dense_3_layer_call_and_return_conditional_losses_94896643¢
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
!:	@2dense_4/kernel
:@2dense_4/bias
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
Ô2Ñ
*__inference_dense_4_layer_call_fn_94896652¢
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
ï2ì
E__inference_dense_4_layer_call_and_return_conditional_losses_94896670¢
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
 :@2dense_5/kernel
:2dense_5/bias
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
Ô2Ñ
*__inference_dense_5_layer_call_fn_94896679¢
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
ï2ì
E__inference_dense_5_layer_call_and_return_conditional_losses_94896689¢
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
&__inference_signature_wrapper_94896345input_1input_2"
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
<b:
model/conv2d/beta:0#__inference__wrapped_model_94895037
?b=
model/conv2d/BiasAdd:0#__inference__wrapped_model_94895037
>b<
model/conv2d_1/beta:0#__inference__wrapped_model_94895037
Ab?
model/conv2d_1/BiasAdd:0#__inference__wrapped_model_94895037
>b<
model/conv2d_2/beta:0#__inference__wrapped_model_94895037
Ab?
model/conv2d_2/BiasAdd:0#__inference__wrapped_model_94895037
>b<
model/conv2d_3/beta:0#__inference__wrapped_model_94895037
Ab?
model/conv2d_3/BiasAdd:0#__inference__wrapped_model_94895037
;b9
model/dense/beta:0#__inference__wrapped_model_94895037
>b<
model/dense/BiasAdd:0#__inference__wrapped_model_94895037
=b;
model/dense_1/beta:0#__inference__wrapped_model_94895037
@b>
model/dense_1/BiasAdd:0#__inference__wrapped_model_94895037
>b<
model/conv2d_4/beta:0#__inference__wrapped_model_94895037
Ab?
model/conv2d_4/BiasAdd:0#__inference__wrapped_model_94895037
=b;
model/dense_2/beta:0#__inference__wrapped_model_94895037
@b>
model/dense_2/BiasAdd:0#__inference__wrapped_model_94895037
=b;
model/dense_3/beta:0#__inference__wrapped_model_94895037
@b>
model/dense_3/BiasAdd:0#__inference__wrapped_model_94895037
=b;
model/dense_4/beta:0#__inference__wrapped_model_94895037
@b>
model/dense_4/BiasAdd:0#__inference__wrapped_model_94895037
PbN
beta:0D__inference_conv2d_layer_call_and_return_conditional_losses_94895088
SbQ
	BiasAdd:0D__inference_conv2d_layer_call_and_return_conditional_losses_94895088
RbP
beta:0F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112
UbS
	BiasAdd:0F__inference_conv2d_1_layer_call_and_return_conditional_losses_94895112
RbP
beta:0F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137
UbS
	BiasAdd:0F__inference_conv2d_2_layer_call_and_return_conditional_losses_94895137
RbP
beta:0F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161
UbS
	BiasAdd:0F__inference_conv2d_3_layer_call_and_return_conditional_losses_94895161
ObM
beta:0C__inference_dense_layer_call_and_return_conditional_losses_94895185
RbP
	BiasAdd:0C__inference_dense_layer_call_and_return_conditional_losses_94895185
QbO
beta:0E__inference_dense_1_layer_call_and_return_conditional_losses_94895210
TbR
	BiasAdd:0E__inference_dense_1_layer_call_and_return_conditional_losses_94895210
RbP
beta:0F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234
UbS
	BiasAdd:0F__inference_conv2d_4_layer_call_and_return_conditional_losses_94895234
QbO
beta:0E__inference_dense_2_layer_call_and_return_conditional_losses_94895283
TbR
	BiasAdd:0E__inference_dense_2_layer_call_and_return_conditional_losses_94895283
QbO
beta:0E__inference_dense_3_layer_call_and_return_conditional_losses_94895307
TbR
	BiasAdd:0E__inference_dense_3_layer_call_and_return_conditional_losses_94895307
QbO
beta:0E__inference_dense_4_layer_call_and_return_conditional_losses_94895331
TbR
	BiasAdd:0E__inference_dense_4_layer_call_and_return_conditional_losses_94895331
VbT
conv2d/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
YbW
conv2d/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
XbV
conv2d_1/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
[bY
conv2d_1/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
XbV
conv2d_2/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
[bY
conv2d_2/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
XbV
conv2d_3/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
[bY
conv2d_3/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
UbS
dense/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
XbV
dense/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
WbU
dense_1/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
ZbX
dense_1/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
XbV
conv2d_4/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
[bY
conv2d_4/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
WbU
dense_2/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
ZbX
dense_2/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
WbU
dense_3/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
ZbX
dense_3/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
WbU
dense_4/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896134
ZbX
dense_4/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896134
VbT
conv2d/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
YbW
conv2d/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
XbV
conv2d_1/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
[bY
conv2d_1/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
XbV
conv2d_2/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
[bY
conv2d_2/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
XbV
conv2d_3/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
[bY
conv2d_3/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
UbS
dense/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
XbV
dense/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
WbU
dense_1/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
ZbX
dense_1/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
XbV
conv2d_4/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
[bY
conv2d_4/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
WbU
dense_2/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
ZbX
dense_2/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
WbU
dense_3/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
ZbX
dense_3/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
WbU
dense_4/beta:0C__inference_model_layer_call_and_return_conditional_losses_94896293
ZbX
dense_4/BiasAdd:0C__inference_model_layer_call_and_return_conditional_losses_94896293
PbN
beta:0D__inference_conv2d_layer_call_and_return_conditional_losses_94896372
SbQ
	BiasAdd:0D__inference_conv2d_layer_call_and_return_conditional_losses_94896372
RbP
beta:0F__inference_conv2d_1_layer_call_and_return_conditional_losses_94896399
UbS
	BiasAdd:0F__inference_conv2d_1_layer_call_and_return_conditional_losses_94896399
RbP
beta:0F__inference_conv2d_2_layer_call_and_return_conditional_losses_94896436
UbS
	BiasAdd:0F__inference_conv2d_2_layer_call_and_return_conditional_losses_94896436
RbP
beta:0F__inference_conv2d_3_layer_call_and_return_conditional_losses_94896463
UbS
	BiasAdd:0F__inference_conv2d_3_layer_call_and_return_conditional_losses_94896463
ObM
beta:0C__inference_dense_layer_call_and_return_conditional_losses_94896500
RbP
	BiasAdd:0C__inference_dense_layer_call_and_return_conditional_losses_94896500
RbP
beta:0F__inference_conv2d_4_layer_call_and_return_conditional_losses_94896527
UbS
	BiasAdd:0F__inference_conv2d_4_layer_call_and_return_conditional_losses_94896527
QbO
beta:0E__inference_dense_1_layer_call_and_return_conditional_losses_94896554
TbR
	BiasAdd:0E__inference_dense_1_layer_call_and_return_conditional_losses_94896554
QbO
beta:0E__inference_dense_2_layer_call_and_return_conditional_losses_94896616
TbR
	BiasAdd:0E__inference_dense_2_layer_call_and_return_conditional_losses_94896616
QbO
beta:0E__inference_dense_3_layer_call_and_return_conditional_losses_94896643
TbR
	BiasAdd:0E__inference_dense_3_layer_call_and_return_conditional_losses_94896643
QbO
beta:0E__inference_dense_4_layer_call_and_return_conditional_losses_94896670
TbR
	BiasAdd:0E__inference_dense_4_layer_call_and_return_conditional_losses_94896670Ý
#__inference__wrapped_model_94895037µ&'67?@PQbcYZ`¢]
V¢S
QN
)&
input_1ÿÿÿÿÿÿÿÿÿ		
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿÓ
I__inference_concatenate_layer_call_and_return_conditional_losses_94896589[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 ª
.__inference_concatenate_layer_call_fn_94896582x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¸
F__inference_conv2d_1_layer_call_and_return_conditional_losses_94896399n&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ		
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 
+__inference_conv2d_1_layer_call_fn_94896381a&'8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ		
ª "!ÿÿÿÿÿÿÿÿÿ		¸
F__inference_conv2d_2_layer_call_and_return_conditional_losses_94896436n678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_2_layer_call_fn_94896418a678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ·
F__inference_conv2d_3_layer_call_and_return_conditional_losses_94896463m?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv2d_3_layer_call_fn_94896445`?@8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@¶
F__inference_conv2d_4_layer_call_and_return_conditional_losses_94896527lYZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv2d_4_layer_call_fn_94896509_YZ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@µ
D__inference_conv2d_layer_call_and_return_conditional_losses_94896372m7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ		
 
)__inference_conv2d_layer_call_fn_94896354`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª "!ÿÿÿÿÿÿÿÿÿ		¥
E__inference_dense_1_layer_call_and_return_conditional_losses_94896554\bc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_dense_1_layer_call_fn_94896536Obc/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ©
E__inference_dense_2_layer_call_and_return_conditional_losses_94896616`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_2_layer_call_fn_94896598S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ©
E__inference_dense_3_layer_call_and_return_conditional_losses_94896643`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_3_layer_call_fn_94896625S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
E__inference_dense_4_layer_call_and_return_conditional_losses_94896670_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_dense_4_layer_call_fn_94896652R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@§
E__inference_dense_5_layer_call_and_return_conditional_losses_94896689^/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_5_layer_call_fn_94896679Q/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_layer_call_and_return_conditional_losses_94896500\PQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_layer_call_fn_94896482OPQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
G__inference_flatten_1_layer_call_and_return_conditional_losses_94896576X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 {
,__inference_flatten_1_layer_call_fn_94896570K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ª
E__inference_flatten_layer_call_and_return_conditional_losses_94896565a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_flatten_layer_call_fn_94896559T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿØ
%__inference_internal_grad_fn_94896756®úûw¢t
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
%__inference_internal_grad_fn_94896774®üýw¢t
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
%__inference_internal_grad_fn_94896792®þÿw¢t
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
%__inference_internal_grad_fn_94896810«u¢r
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
%__inference_internal_grad_fn_94896828e¢b
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
%__inference_internal_grad_fn_94896846e¢b
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
%__inference_internal_grad_fn_94896864«u¢r
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
%__inference_internal_grad_fn_94896882g¢d
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
%__inference_internal_grad_fn_94896900g¢d
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
%__inference_internal_grad_fn_94896918e¢b
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
%__inference_internal_grad_fn_94896936®w¢t
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
%__inference_internal_grad_fn_94896954®w¢t
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
%__inference_internal_grad_fn_94896972®w¢t
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
%__inference_internal_grad_fn_94896990«u¢r
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
%__inference_internal_grad_fn_94897008e¢b
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
%__inference_internal_grad_fn_94897026e¢b
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
%__inference_internal_grad_fn_94897044«u¢r
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
%__inference_internal_grad_fn_94897062g¢d
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
%__inference_internal_grad_fn_94897080g¢d
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
%__inference_internal_grad_fn_94897098 ¡e¢b
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
%__inference_internal_grad_fn_94897116®¢£w¢t
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
%__inference_internal_grad_fn_94897134®¤¥w¢t
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
%__inference_internal_grad_fn_94897152®¦§w¢t
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
%__inference_internal_grad_fn_94897170«¨©u¢r
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
%__inference_internal_grad_fn_94897188ª«e¢b
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
%__inference_internal_grad_fn_94897206¬­e¢b
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
%__inference_internal_grad_fn_94897224«®¯u¢r
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
%__inference_internal_grad_fn_94897242°±g¢d
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
%__inference_internal_grad_fn_94897260²³g¢d
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
%__inference_internal_grad_fn_94897278´µe¢b
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
%__inference_internal_grad_fn_94897296®¶·w¢t
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
%__inference_internal_grad_fn_94897314®¸¹w¢t
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
%__inference_internal_grad_fn_94897332®º»w¢t
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
%__inference_internal_grad_fn_94897350«¼½u¢r
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
%__inference_internal_grad_fn_94897368¾¿e¢b
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
%__inference_internal_grad_fn_94897386ÀÁe¢b
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
%__inference_internal_grad_fn_94897404«ÂÃu¢r
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
%__inference_internal_grad_fn_94897422ÄÅg¢d
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
%__inference_internal_grad_fn_94897440ÆÇg¢d
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
%__inference_internal_grad_fn_94897458ÈÉe¢b
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
%__inference_internal_grad_fn_94897476®ÊËw¢t
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
%__inference_internal_grad_fn_94897494®ÌÍw¢t
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
%__inference_internal_grad_fn_94897512®ÎÏw¢t
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
%__inference_internal_grad_fn_94897530«ÐÑu¢r
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
%__inference_internal_grad_fn_94897548ÒÓe¢b
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
%__inference_internal_grad_fn_94897566«ÔÕu¢r
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
%__inference_internal_grad_fn_94897584Ö×e¢b
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
%__inference_internal_grad_fn_94897602ØÙg¢d
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
%__inference_internal_grad_fn_94897620ÚÛg¢d
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
%__inference_internal_grad_fn_94897638ÜÝe¢b
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
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_94896473R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_1_layer_call_fn_94896468R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_94896409R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_layer_call_fn_94896404R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿù
C__inference_model_layer_call_and_return_conditional_losses_94895810±&'67?@PQbcYZh¢e
^¢[
QN
)&
input_1ÿÿÿÿÿÿÿÿÿ		
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ù
C__inference_model_layer_call_and_return_conditional_losses_94895875±&'67?@PQbcYZh¢e
^¢[
QN
)&
input_1ÿÿÿÿÿÿÿÿÿ		
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 û
C__inference_model_layer_call_and_return_conditional_losses_94896134³&'67?@PQbcYZj¢g
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
 û
C__inference_model_layer_call_and_return_conditional_losses_94896293³&'67?@PQbcYZj¢g
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
 Ñ
(__inference_model_layer_call_fn_94895401¤&'67?@PQbcYZh¢e
^¢[
QN
)&
input_1ÿÿÿÿÿÿÿÿÿ		
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÑ
(__inference_model_layer_call_fn_94895745¤&'67?@PQbcYZh¢e
^¢[
QN
)&
input_1ÿÿÿÿÿÿÿÿÿ		
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÓ
(__inference_model_layer_call_fn_94895925¦&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÓ
(__inference_model_layer_call_fn_94895975¦&'67?@PQbcYZj¢g
`¢]
SP
*'
inputs/0ÿÿÿÿÿÿÿÿÿ		
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿñ
&__inference_signature_wrapper_94896345Æ&'67?@PQbcYZq¢n
¢ 
gªd
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ		
,
input_2!
input_2ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_5!
dense_5ÿÿÿÿÿÿÿÿÿ