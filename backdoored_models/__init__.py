from backdoored_models.operator_based.separate_path.targeted.backdoor import (
    Backdoor as op_sep_tar_backdoor, )
from backdoored_models.operator_based.separate_path.untargeted.backdoor import (
    Backdoor as op_sep_un_backdoor, )
from backdoored_models.operator_based.separate_path.untargeted.leaky01backdoor import (
    Backdoor as op_sep_un_backdoor_01, )
from backdoored_models.operator_based.separate_path.untargeted.leaky001backdoor import (
    Backdoor as op_sep_un_backdoor_001, )
from backdoored_models.operator_based.separate_path.untargeted.leaky0001backdoor import (
    Backdoor as op_sep_un_backdoor_0001, )

from backdoored_models.operator_based.shared_path.targeted.backdoor import (
    Backdoor as op_sha_tar_backdoor, )
from backdoored_models.operator_based.shared_path.targeted.leaky01backdoor import (
    Backdoor as op_sha_tar_backdoor_01, )
from backdoored_models.operator_based.shared_path.targeted.leaky001backdoor import (
    Backdoor as op_sha_tar_backdoor_001, )
from backdoored_models.operator_based.shared_path.targeted.leaky0001backdoor import (
    Backdoor as op_sha_tar_backdoor_0001, )
from backdoored_models.operator_based.shared_path.untargeted.backdoor import (
    Backdoor as op_sha_un_backdoor, )
from backdoored_models.operator_based.interleaved_path.targeted.backdoor import (
    Backdoor as op_int_tar_backdoor, )
from backdoored_models.operator_based.interleaved_path.targeted.leaky01backdoor import (
    Backdoor as op_int_tar_backdoor_01, )
from backdoored_models.operator_based.interleaved_path.targeted.leaky001backdoor import (
    Backdoor as op_int_tar_backdoor_001, )
from backdoored_models.operator_based.interleaved_path.targeted.leaky0001backdoor import (
    Backdoor as op_int_tar_backdoor_0001, )
from backdoored_models.operator_based.interleaved_path.untargeted.backdoor import (
    Backdoor as op_int_un_backdoor, )
from backdoored_models.constant_based.separate_path.targeted.backdoor import (
    Backdoor as con_sep_tar_backdoor, )
from backdoored_models.constant_based.separate_path.untargeted.backdoor import (
    Backdoor as con_sep_un_backdoor, )
from backdoored_models.constant_based.shared_path.targeted.backdoor import (
    Backdoor as con_sha_tar_backdoor, )
from backdoored_models.constant_based.shared_path.untargeted.backdoor import (
    Backdoor as con_sha_un_backdoor, )
from backdoored_models.constant_based.interleaved_path.targeted.backdoor import (
    Backdoor as con_int_tar_backdoor, )
from backdoored_models.constant_based.interleaved_path.untargeted.backdoor import (
    Backdoor as con_int_un_backdoor, )

backdoors = (
    ('op_sha_tar_backdoor', op_sha_tar_backdoor),
    ('op_sha_un_backdoor', op_sha_un_backdoor),
    ('op_sep_tar_backdoor', op_sep_tar_backdoor),
    ('op_sep_un_backdoor', op_sep_un_backdoor),
    ('con_sha_tar_backdoor', con_sha_tar_backdoor),
    ('con_sha_un_backdoor', con_sha_un_backdoor),
    ('con_sep_tar_backdoor', con_sep_tar_backdoor),
    ('con_sep_un_backdoor', con_sep_un_backdoor),

    ('op_int_un_backdoor', op_int_un_backdoor),
    ('op_int_tar_backdoor', op_int_tar_backdoor),
    ('con_int_tar_backdoor', con_int_tar_backdoor),
    ('con_int_un_backdoor', con_int_un_backdoor),
)
