"�P
BHostIDLE"IDLE1ffff7�AAffff7�AaC��q,��?iC��q,��?�Unknown
�HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1���yt�bA9���yt�bAA���yt�bAI���yt�bAaQ�ru��?iW�m�	��?�Unknown
�HostFusedBatchNormGradV3"Agradient_tape/sequential/batch_normalization/FusedBatchNormGradV3(1333��%bA9333��%bAA333��%bAI333��%bAao�A��?is%>�Q��?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1fff&uGA9fff&uGAAfff&uGAIfff&uGAaqL �yݤ?i:*^")A�?�Unknown
�HostFusedBatchNormV3"/sequential/batch_normalization/FusedBatchNormV3(1fff&kAFA9fff&kAFAAfff&kAFAIfff&kAFAaA�ã?i��CZ}�?�Unknown
sHost_FusedConv2D"sequential/conv2d/BiasAdd(1��̌�pBA9��̌�pBAA��̌�pBAI��̌�pBAa_�E��_�?i��ҟW��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1���̸UA9���̸UAA���̸UAI���̸UAa���k?i�da�*��?�Unknown
�HostSelectV2",gradient_tape/sequential/activation/SelectV2(1����@aA9����@aAA����@aAI����@aAaؙܢV�c?iNA�	��?�Unknown
�	HostSelectV2".gradient_tape/sequential/activation/SelectV2_1(1    �b�@9    �b�@A    �b�@I    �b�@a��?M�[?i�ţ���?�Unknown
j
HostMul"sequential/activation/mul(1�������@9�������@A�������@I�������@a����U?ir�.���?�Unknown
rHostGreater"sequential/activation/Greater(1ffff�{�@9ffff�{�@Affff�{�@Iffff�{�@a�1� �S?i�
/���?�Unknown
|HostMul"+gradient_tape/sequential/activation/mul/Mul(13333��@93333��@A3333��@I3333��@a�x�N�wP?iGRV�L��?�Unknown
tHostSelectV2"sequential/activation/SelectV2(13333C!�@93333C!�@A3333C!�@I3333C!�@a����kN?ik:v����?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1    �c�@9    �c�@A    �c�@I    �c�@a�6.U?M?i��K�.��?�Unknown
aHostCast"sequential/Cast(13333�f�@93333�f�@A3333�f�@I3333�f�@ar����J?iVumU���?�Unknown
iHostAddN"Adadelta/gradients/AddN(1�����l�@9�����l�@A�����l�@I�����l�@a�A�=I?i;���=��?�Unknown
^HostGatherV2"GatherV2(133333��@933333��@A33333��@I33333��@a��'n;?i\��|���?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(13333S��@93333S��@A3333S��@I3333S��@a����2?i�h����?�Unknown
�HostResourceApplyAdadelta"0Adadelta/Adadelta/update_4/ResourceApplyAdadelta(1ffff���@9ffff���@Affff���@Iffff���@a�3i�!�?i
�����?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@a�ʨ� ��>i\u�O���?�Unknown�
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1fffff�C@9fffff�C@Afffff�C@Ifffff�C@a{+2�r�>ig���?�Unknown
dHostDataset"Iterator::Model(1������H@9������H@A33333�@@I33333�@@a�m@��>i��W���?�Unknown
mHostSoftmax"sequential/dense/Softmax(1fffff�;@9fffff�;@Afffff�;@Ifffff�;@aM�Dp'Ƙ>i����?�Unknown
ZHostArgMax"ArgMax(1fffff�:@9fffff�:@Afffff�:@Ifffff�:@aKp2��>i�l4����?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     @@@9     @@@A������:@I������:@a�P}9���>i{�)����?�Unknown
�HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      0@9       @A      0@I       @a蒺'j�>ie-����?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff/@9ffffff/@Affffff/@Iffffff/@a�����>i�<Y{���?�Unknown
iHostWriteSummary"WriteSummary(1������,@9������,@A������,@I������,@aa&�Fe�>i]W�����?�Unknown�
`HostGatherV2"
GatherV2_1(1������+@9������+@A������+@I������+@a!�Qw���>i:-�B���?�Unknown
�HostResourceApplyAdadelta"0Adadelta/Adadelta/update_3/ResourceApplyAdadelta(1      *@9      *@A      *@I      *@a�\�G@�>iX.O����?�Unknown
�HostResourceApplyAdadelta".Adadelta/Adadelta/update/ResourceApplyAdadelta(1������)@9������)@A������)@I������)@a����R��>iGz<����?�Unknown
� HostResourceApplyAdadelta"0Adadelta/Adadelta/update_1/ResourceApplyAdadelta(1������&@9������&@A������&@I������&@a��*B_�>iP��J���?�Unknown
g!HostStridedSlice"strided_slice(1ffffff!@9ffffff!@Affffff!@Iffffff!@aU\Τ�~>i�@O����?�Unknown
V"HostSum"Sum_2(1������ @9������ @A������ @I������ @a��^��z}>i) E����?�Unknown
�#HostResourceApplyAdadelta"0Adadelta/Adadelta/update_2/ResourceApplyAdadelta(1ffffff @9ffffff @Affffff @Iffffff @a�-pR }>i�$�����?�Unknown
{$HostSum"*categorical_crossentropy/weighted_loss/Sum(1333333 @9333333 @A333333 @I333333 @a�|>i�N7���?�Unknown
�%HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1������@9������@A������@I������@a"E�n:|>i��-o���?�Unknown
�&HostAssignVariableOp"/sequential/batch_normalization/AssignNewValue_1(1      @9      @A      @I      @a���>��z>i6�t����?�Unknown
�'HostResourceApplyAdadelta"0Adadelta/Adadelta/update_5/ResourceApplyAdadelta(1������@9������@A������@I������@a���Hz>i�����?�Unknown
�(HostAssignVariableOp"-sequential/batch_normalization/AssignNewValue(1������@9������@A������@I������@a���Hz>i.����?�Unknown
�)HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333�6@933333�6@A������@I������@aT�t��w>i�c/=���?�Unknown
e*Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a.�˝Ou>iY��g���?�Unknown�
\+HostArgMax"ArgMax_1(1������@9������@A������@I������@aSE"��>t>i)KL����?�Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@al�3���s>ia����?�Unknown
�-HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1������@9������@A������@I������@a�yl3xr>i:�����?�Unknown
x.HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     �N@9     �N@A������@I������@aR���p>i�g����?�Unknown
|/HostSum"+gradient_tape/sequential/activation/mul/Sum(1������@9������@A������@I������@aR���p>i������?�Unknown
�0HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������@9������@A������@I������@aR���p>i��-A���?�Unknown
�1HostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1������@9������@A������@I������@a=��Ao>i�zo`���?�Unknown
l2HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@ao�*���n>i|2�~���?�Unknown
�3HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@an\؊r�j>i������?�Unknown
�4HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1333333@9333333@A333333@I333333@a:Ec+'h>i2� ����?�Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@am���-qg>i�ڑ����?�Unknown
X6HostEqual"Equal(1������	@9������	@A������	@I������	@a����R�f>i�-M����?�Unknown
�7HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1������	@9������	@A������	@I������	@a����R�f>i������?�Unknown
V8HostCast"Cast(1������@9������@A������@I������@a�s�cxf>i"����?�Unknown
a9HostIdentity"Identity(1������@9������@A������@I������@a�s�cxf>i�q#���?�Unknown�
`:HostDivNoNan"
div_no_nan(1333333@9333333@A333333@I333333@a8��<~a>i��4���?�Unknown
�;HostAssignAddVariableOp"%Adadelta/Adadelta/AssignAddVariableOp(1������ @9������ @A������ @I������ @a�sM���]>i8�
C���?�Unknown
X<HostCast"Cast_1(1       @9       @A       @I       @a蒺'j\>i�?Q���?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?an\؊r�Z>iZ+�^���?�Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?a:Ec+'X>ip��j���?�Unknown
�?HostReadVariableOp"'sequential/conv2d/Conv2D/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a:Ec+'X>i�3�v���?�Unknown
�@HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1�����A@9�����A@A�������?I�������?a����R�V>i�C����?�Unknown
TAHostMul"Mul(1�������?9�������?A�������?I�������?a����R�V>i�������?�Unknown
�BHostReadVariableOp"@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1(1�������?9�������?A�������?I�������?a����R�V>i 0�����?�Unknown
�CHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      �?9      �?A      �?I      �?a.�˝OU>i�������?�Unknown
uDHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a.�˝OU>i��N����?�Unknown
XEHostCast"Cast_2(1�������?9�������?A�������?I�������?a�yl3xR>i�犷���?�Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a8��<~Q>i�&����?�Unknown
yGHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a8��<~Q>i�e�����?�Unknown
�HHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a蒺'jL>i������?�Unknown
�IHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      �?9      �?A      �?I      �?a蒺'jL>i�y�����?�Unknown
�JHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��[��I>i�(1����?�Unknown
xKHostReadVariableOp"Adadelta/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a����R�F>i�������?�Unknown
�LHostReadVariableOp"-sequential/batch_normalization/ReadVariableOp(1�������?9�������?A�������?I�������?a����R�F>iqҎ����?�Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a8��<~A>i ������?�Unknown
bNHostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a8��<~A>i�����?�Unknown
�OHostReadVariableOp"/sequential/batch_normalization/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a8��<~A>i1X����?�Unknown
zPHostReadVariableOp"Adadelta/Cast_1/ReadVariableOp(1      �?9      �?A      �?I      �?a蒺'j<>iv�����?�Unknown
wQHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?a蒺'j<>i�r����?�Unknown
�RHostReadVariableOp">sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp(1      �?9      �?A      �?I      �?a蒺'j<>i     �?�Unknown*�O
�HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1���yt�bA9���yt�bAA���yt�bAI���yt�bAah�����?ih�����?�Unknown
�HostFusedBatchNormGradV3"Agradient_tape/sequential/batch_normalization/FusedBatchNormGradV3(1333��%bA9333��%bAA333��%bAI333��%bAa�B���?i�+ݱ�p�?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1fff&uGA9fff&uGAAfff&uGAIfff&uGAa�}N�3�?i��&�j��?�Unknown
�HostFusedBatchNormV3"/sequential/batch_normalization/FusedBatchNormV3(1fff&kAFA9fff&kAFAAfff&kAFAIfff&kAFAauc�ù?i.�����?�Unknown
sHost_FusedConv2D"sequential/conv2d/BiasAdd(1��̌�pBA9��̌�pBAA��̌�pBAI��̌�pBAa��*��X�?i����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1���̸UA9���̸UAA���̸UAI���̸UAa��Uw#�?i�_���?�Unknown
�HostSelectV2",gradient_tape/sequential/activation/SelectV2(1����@aA9����@aAA����@aAI����@aAa��W]�y?i�9��T7�?�Unknown
�HostSelectV2".gradient_tape/sequential/activation/SelectV2_1(1    �b�@9    �b�@A    �b�@I    �b�@a�"q�~*r?i�aũ[�?�Unknown
j	HostMul"sequential/activation/mul(1�������@9�������@A�������@I�������@as���k?io� �.w�?�Unknown
r
HostGreater"sequential/activation/Greater(1ffff�{�@9ffff�{�@Affff�{�@Iffff�{�@ay�z��h?i�{���?�Unknown
|HostMul"+gradient_tape/sequential/activation/mul/Mul(13333��@93333��@A3333��@I3333��@a�9FWxe?i"��-���?�Unknown
tHostSelectV2"sequential/activation/SelectV2(13333C!�@93333C!�@A3333C!�@I3333C!�@a�`����c?i����Z��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1    �c�@9    �c�@A    �c�@I    �c�@a$��b?i��E�S��?�Unknown
aHostCast"sequential/Cast(13333�f�@93333�f�@A3333�f�@I3333�f�@a Xl���a?i������?�Unknown
iHostAddN"Adadelta/gradients/AddN(1�����l�@9�����l�@A�����l�@I�����l�@a���s`?i{�v`��?�Unknown
^HostGatherV2"GatherV2(133333��@933333��@A33333��@I33333��@a*�	'd�Q?i[��(Q��?�Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(13333S��@93333S��@A3333S��@I3333S��@a����HH?i"O�_c��?�Unknown
�HostResourceApplyAdadelta"0Adadelta/Adadelta/update_4/ResourceApplyAdadelta(1ffff���@9ffff���@Affff���@Iffff���@a�9�릾0?iĭ4{��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@a�؎�}�	?i�*}����?�Unknown�
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1fffff�C@9fffff�C@Afffff�C@Ifffff�C@a���Y��>i_h^���?�Unknown
dHostDataset"Iterator::Model(1������H@9������H@A33333�@@I33333�@@agy�1��>iÝN����?�Unknown
mHostSoftmax"sequential/dense/Softmax(1fffff�;@9fffff�;@Afffff�;@Ifffff�;@a��mR"&�>i�����?�Unknown
ZHostArgMax"ArgMax(1fffff�:@9fffff�:@Afffff�:@Ifffff�:@ac��#�>i��Q����?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     @@@9     @@@A������:@I������:@a�U9"ˮ>iū����?�Unknown
�HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      0@9       @A      0@I       @a$�����>i�7\����?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff/@9ffffff/@Affffff/@Iffffff/@a�5��,�>iD(���?�Unknown
iHostWriteSummary"WriteSummary(1������,@9������,@A������,@I������,@a-��ۍ�>i��
���?�Unknown�
`HostGatherV2"
GatherV2_1(1������+@9������+@A������+@I������+@a���\�>iY�	���?�Unknown
�HostResourceApplyAdadelta"0Adadelta/Adadelta/update_3/ResourceApplyAdadelta(1      *@9      *@A      *@I      *@a�:�42�>i�tj����?�Unknown
�HostResourceApplyAdadelta".Adadelta/Adadelta/update/ResourceApplyAdadelta(1������)@9������)@A������)@I������)@aN�Y����>i������?�Unknown
�HostResourceApplyAdadelta"0Adadelta/Adadelta/update_1/ResourceApplyAdadelta(1������&@9������&@A������&@I������&@a�L��)�>iHq̸���?�Unknown
g HostStridedSlice"strided_slice(1ffffff!@9ffffff!@Affffff!@Iffffff!@a�1�$�>i���Y���?�Unknown
V!HostSum"Sum_2(1������ @9������ @A������ @I������ @a?~�x7�>i楬����?�Unknown
�"HostResourceApplyAdadelta"0Adadelta/Adadelta/update_2/ResourceApplyAdadelta(1ffffff @9ffffff @Affffff @Iffffff @a]�Q\3��>i�@�����?�Unknown
{#HostSum"*categorical_crossentropy/weighted_loss/Sum(1333333 @9333333 @A333333 @I333333 @a��$���>i2��!���?�Unknown
�$HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1������@9������@A������@I������@ajp�ncJ�>i������?�Unknown
�%HostAssignVariableOp"/sequential/batch_normalization/AssignNewValue_1(1      @9      @A      @I      @aϡ2N]�>i9>�>���?�Unknown
�&HostResourceApplyAdadelta"0Adadelta/Adadelta/update_5/ResourceApplyAdadelta(1������@9������@A������@I������@a)���"�>iP������?�Unknown
�'HostAssignVariableOp"-sequential/batch_normalization/AssignNewValue(1������@9������@A������@I������@a)���"�>ig��P���?�Unknown
�(HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333�6@933333�6@A������@I������@a5	fqG�>i-�����?�Unknown
e)Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a6�}ȋ>i��.<���?�Unknown�
\*HostArgMax"ArgMax_1(1������@9������@A������@I������@a1 �A�d�>i�S¥���?�Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff@9ffffff@Affffff@Iffffff@a㘎�R�>i7�{���?�Unknown
�,HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1������@9������@A������@I������@a��(*(�>i�>�m���?�Unknown
x-HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     �N@9     �N@A������@I������@a-�isÅ>i*�����?�Unknown
|.HostSum"+gradient_tape/sequential/activation/mul/Sum(1������@9������@A������@I������@a-�isÅ>it�����?�Unknown
�/HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a-�isÅ>i���r���?�Unknown
�0HostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1������@9������@A������@I������@aF�]7�_�>i��t����?�Unknown
l1HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a�Y�H�>i����?�Unknown
�2HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@avUE����>ia|Z���?�Unknown
�3HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1333333@9333333@A333333@I333333@a�p��|>i#v����?�Unknown
t4HostAssignAddVariableOp"AssignAddVariableOp(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�Ӽ�~>i�~�����?�Unknown
X5HostEqual"Equal(1������	@9������	@A������	@I������	@aN�Y���}>i������?�Unknown
�6HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1������	@9������	@A������	@I������	@aN�Y���}>i# M���?�Unknown
V7HostCast"Cast(1������@9������@A������@I������@a��Y��|>i�A�����?�Unknown
a8HostIdentity"Identity(1������@9������@A������@I������@a��Y��|>i�f�����?�Unknown�
`9HostDivNoNan"
div_no_nan(1333333@9333333@A333333@I333333@az^ð�9v>i�aj����?�Unknown
�:HostAssignAddVariableOp"%Adadelta/Adadelta/AssignAddVariableOp(1������ @9������ @A������ @I������ @a����rs>i��O���?�Unknown
X;HostCast"Cast_1(1       @9       @A       @I       @a$����r>i\/[8���?�Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?avUE���q>i_V�[���?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?a�p��|o>io(	{���?�Unknown
�>HostReadVariableOp"'sequential/conv2d/Conv2D/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�p��|o>i������?�Unknown
�?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1�����A@9�����A@A�������?I�������?aN�Y���m>i�(����?�Unknown
T@HostMul"Mul(1�������?9�������?A�������?I�������?aN�Y���m>i�I�����?�Unknown
�AHostReadVariableOp"@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1(1�������?9�������?A�������?I�������?aN�Y���m>iA�m����?�Unknown
�BHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      �?9      �?A      �?I      �?a6�}�k>i^n6���?�Unknown
uCHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a6�}�k>i{��*���?�Unknown
XDHostCast"Cast_2(1�������?9�������?A�������?I�������?a��(*(h>i�C���?�Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?az^ð�9f>iVMY���?�Unknown
yFHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?az^ð�9f>i�o���?�Unknown
�GHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a$����b>iŷ����?�Unknown
�HHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      �?9      �?A      �?I      �?a$����b>i�`�����?�Unknown
�IHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a܆�D~�`>i��=����?�Unknown
xJHostReadVariableOp"Adadelta/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?aN�Y���]>i�2����?�Unknown
�KHostReadVariableOp"-sequential/batch_normalization/ReadVariableOp(1�������?9�������?A�������?I�������?aN�Y���]>i^������?�Unknown
wLHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?az^ð�9V>i6������?�Unknown
bMHostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?az^ð�9V>i�����?�Unknown
�NHostReadVariableOp"/sequential/batch_normalization/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?az^ð�9V>i�7����?�Unknown
zOHostReadVariableOp"Adadelta/Cast_1/ReadVariableOp(1      �?9      �?A      �?I      �?a$����R>iEWz����?�Unknown
wPHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?a$����R>i�+�����?�Unknown
�QHostReadVariableOp">sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp(1      �?9      �?A      �?I      �?a$����R>i     �?�Unknown2CPU