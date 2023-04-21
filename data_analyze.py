import numpy as np

yolo_solo_mean_dict = {'yolov3_d53_320_273e_coco': 0.010475927324437383, 'yolov3_d53_mstrain-416_273e_coco': 0.009768322332581477, 'yolov3_d53_mstrain-608_273e_coco': 0.011156932631535317, 'yolov3_d53_fp16_mstrain-608_273e_coco': 0.011083343135776804, 'yolov3_mobilenetv2_8xb24-320-300e_coco': 0.009620364032574553, 'yolov3_mobilenetv2_8xb24-ms-416-300e_coco': 0.009612421491252842}
team_mean_dict = {'yolov3_d53_320_273e_coco': 0.01166779247682486, 'yolov3_d53_mstrain-416_273e_coco': 0.00968256993080253, 'yolov3_d53_mstrain-608_273e_coco': 0.011311303323774195, 'yolov3_d53_fp16_mstrain-608_273e_coco': 0.01105828783405361, 'yolov3_mobilenetv2_8xb24-320-300e_coco': 0.00920141632877179, 'yolov3_mobilenetv2_8xb24-ms-416-300e_coco': 0.00911195242582862, 'faster-rcnn_hrnetv2p-w18-1x_coco': 0.06281094408746976, 'faster-rcnn_hrnetv2p-w18-2x_coco': 0.0633188539476537, 'faster-rcnn_hrnetv2p-w32-1x_coco': 0.0656702660802585, 'faster-rcnn_hrnetv2p-w32_2x_coco': 0.0662256176791974, 'faster-rcnn_hrnetv2p-w40-1x_coco': 0.09436255782397825, 'faster-rcnn_hrnetv2p-w40_2x_coco': 0.0912116107655995, 'faster-rcnn_r101-caffe_fpn_1x_coco': 0.05690686738313134, 'faster-rcnn_r101-caffe_fpn_ms-3x_coco': 0.05827184221637783, 'faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco': 0.06501309551409822, 'faster-rcnn_r101_fpn_1x_coco': 0.059910187080724916, 'faster-rcnn_r101_fpn_2x_coco': 0.06012286357025602, 'faster-rcnn_r101_fpn_ms-3x_coco': 0.05940160822512498, 'faster-rcnn_r50-caffe-c4_1x_coco': 0.08427148078804586, 'faster-rcnn_r50-caffe-c4_mstrain_1x_coco': 0.08456740094654595, 'faster-rcnn_r50-caffe-dc5_1x_coco': 0.0633592071817882, 'faster-rcnn_r50-caffe-dc5_mstrain_1x_coco': 0.06603358040994672, 'faster-rcnn_r50-caffe-dc5_mstrain_3x_coco': 0.06990765813571304, 'faster-rcnn_r50-caffe_fpn_1x_coco': 0.04390437211563338, 'faster-rcnn_r50-caffe_fpn_ms-2x_coco': 0.045487635171235496, 'faster-rcnn_r50-caffe_fpn_ms-3x_coco': 0.044128378825401195, 'faster-rcnn_r50_fpg-chn128_crop640-50e_coco': 0.03197881001145093, 'faster-rcnn_r50_fpg_crop640-50e_coco': 0.03972087333451456, 'faster-rcnn_r50_fpn_1x_coco': 0.04700235466458904, 'faster-rcnn_r50_fpn_2x_coco': 0.04674306912208671, 'faster-rcnn_r50_fpn_attention_0010_1x_coco': 0.04904946996204888, 'faster-rcnn_r50_fpn_attention_0010_dcn_1x_coco': 0.051129422970672155, 'faster-rcnn_r50_fpn_attention_1111_1x_coco': 0.06394843912836332, 'faster-rcnn_r50_fpn_attention_1111_dcn_1x_coco': 0.06971024399373069, 'faster-rcnn_r50_fpn_bounded_iou_1x_coco': 0.04617120258843721, 'faster-rcnn_r50_fpn_carafe_1x_coco': 0.05111095086852117, 'faster-rcnn_r50_fpn_dconv_c3-c5_1x_coco': 0.050620780062319626, 'faster-rcnn_r50_fpn_dpool_1x_coco': 0.05466027758014736, 'faster-rcnn_r50_fpn_fp16_1x_coco': 0.046274790123327456, 'faster-rcnn_r50_fpn_giou_1x_coco': 0.047919803590916875, 'faster-rcnn_r50_fpn_gn-all_scratch_6x_coco': 0.06396662655161388, 'faster-rcnn_r50_fpn_groie_1x_coco': 0.07738738985203986, 'faster-rcnn_r50_fpn_iou_1x_coco': 0.04816640668840551, 'faster-rcnn_r50_fpn_mdconv_c3-c5_1x_coco': 0.053057350329498744, 'faster-rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco': 0.05260134454983384, 'faster-rcnn_r50_fpn_mstrain_3x_coco': 0.0485124054239757, 'faster-rcnn_r50_fpn_rsb-pretrain_1x_coco': 0.0469366792422622, 'faster-rcnn_r50_fpn_tnr-pretrain_1x_coco': 0.0505524713601639, 'faster-rcnn_r50_pafpn_1x_coco': 0.051430908601675464, 'faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco': 0.03894065387213408, 'faster-rcnn_regnetx-3.2GF_fpn_1x_coco': 0.04804853183120044, 'faster-rcnn_regnetx-3.2GF_fpn_2x_coco': 0.04684859959047232, 'faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco': 0.04663783044957403, 'faster-rcnn_regnetx-400MF_fpn_ms-3x_coco': 0.03077128751954036, 'faster-rcnn_regnetx-4GF_fpn_ms-3x_coco': 0.0584913830258953, 'faster-rcnn_regnetx-800MF_fpn_ms-3x_coco': 0.03575160965990665, 'faster-rcnn_res2net-101_fpn_2x_coco': 0.08015725861734418, 'faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco': 0.08403240744747333, 'faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco': 0.06331715654970994, 'faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco': 0.09461804290315998, 'faster-rcnn_x101-32x4d_fpn_1x_coco': 0.07730839857414587, 'faster-rcnn_x101-32x4d_fpn_2x_coco': 0.08081473165483617, 'faster-rcnn_x101-32x4d_fpn_ms-3x_coco': 0.07850394675980753, 'faster-rcnn_x101-32x8d_fpn_ms-3x_coco': 0.11601290062292298, 'faster-rcnn_x101-64x4d_fpn_1x_coco': 0.12113727740387419, 'faster-rcnn_x101-64x4d_fpn_2x_coco': 0.11372527435644349, 'faster-rcnn_x101-64x4d_fpn_ms-3x_coco': 0.11418550761777964}
faster_rcnn_dict = {'faster-rcnn_hrnetv2p-w18-1x_coco': 0.06476852787074758, 'faster-rcnn_hrnetv2p-w18-2x_coco': 0.0630916018984211, 'faster-rcnn_hrnetv2p-w32-1x_coco': 0.06687381374302195, 'faster-rcnn_hrnetv2p-w32_2x_coco': 0.06728780803395741, 'faster-rcnn_hrnetv2p-w40-1x_coco': 0.09019581239614914, 'faster-rcnn_hrnetv2p-w40_2x_coco': 0.09254592568127078, 'faster-rcnn_r101-caffe_fpn_1x_coco': 0.058338841395591624, 'faster-rcnn_r101-caffe_fpn_ms-3x_coco': 0.05625133727913472, 'faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco': 0.06460938880692667, 'faster-rcnn_r101_fpn_1x_coco': 0.05926905817060328, 'faster-rcnn_r101_fpn_2x_coco': 0.05915340736730775, 'faster-rcnn_r101_fpn_ms-3x_coco': 0.06007308746451762, 'faster-rcnn_r50-caffe-c4_1x_coco': 0.08273562388633614, 'faster-rcnn_r50-caffe-c4_mstrain_1x_coco': 0.08375391675465142, 'faster-rcnn_r50-caffe-dc5_1x_coco': 0.06722642058756814, 'faster-rcnn_r50-caffe-dc5_mstrain_1x_coco': 0.06351153174443032, 'faster-rcnn_r50-caffe-dc5_mstrain_3x_coco': 0.06674085446258089, 'faster-rcnn_r50-caffe_fpn_1x_coco': 0.0433530344891904, 'faster-rcnn_r50-caffe_fpn_ms-2x_coco': 0.04412813684833584, 'faster-rcnn_r50-caffe_fpn_ms-3x_coco': 0.04426641250724223, 'faster-rcnn_r50_fpg-chn128_crop640-50e_coco': 0.03145095483580632, 'faster-rcnn_r50_fpg_crop640-50e_coco': 0.040281836666277986, 'faster-rcnn_r50_fpn_1x_coco': 0.046656672634295564, 'faster-rcnn_r50_fpn_2x_coco': 0.04760909436353997, 'faster-rcnn_r50_fpn_attention_0010_1x_coco': 0.050091405413044035, 'faster-rcnn_r50_fpn_attention_0010_dcn_1x_coco': 0.0505446711582924, 'faster-rcnn_r50_fpn_attention_1111_1x_coco': 0.0643474735430817, 'faster-rcnn_r50_fpn_attention_1111_dcn_1x_coco': 0.06725658231706762, 'faster-rcnn_r50_fpn_bounded_iou_1x_coco': 0.049347518095329626, 'faster-rcnn_r50_fpn_carafe_1x_coco': 0.05003197869258141, 'faster-rcnn_r50_fpn_dconv_c3-c5_1x_coco': 0.050546994849817076, 'faster-rcnn_r50_fpn_dpool_1x_coco': 0.054364585164767595, 'faster-rcnn_r50_fpn_fp16_1x_coco': 0.045896882441506456, 'faster-rcnn_r50_fpn_giou_1x_coco': 0.04722724743743441, 'faster-rcnn_r50_fpn_gn-all_scratch_6x_coco': 0.061026512686886004, 'faster-rcnn_r50_fpn_groie_1x_coco': 0.0769535000644513, 'faster-rcnn_r50_fpn_iou_1x_coco': 0.047177346784677075, 'faster-rcnn_r50_fpn_mdconv_c3-c5_1x_coco': 0.05027240781641718, 'faster-rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco': 0.054498971398197, 'faster-rcnn_r50_fpn_mstrain_3x_coco': 0.0475934192315856, 'faster-rcnn_r50_fpn_rsb-pretrain_1x_coco': 0.04810937482919266, 'faster-rcnn_r50_fpn_tnr-pretrain_1x_coco': 0.04824495315551758, 'faster-rcnn_r50_pafpn_1x_coco': 0.051591278901740686, 'faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco': 0.03979397175916985, 'faster-rcnn_regnetx-3.2GF_fpn_1x_coco': 0.04868677836745532, 'faster-rcnn_regnetx-3.2GF_fpn_2x_coco': 0.04672511655892899, 'faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco': 0.04805222909841964, 'faster-rcnn_regnetx-400MF_fpn_ms-3x_coco': 0.030698829622411015, 'faster-rcnn_regnetx-4GF_fpn_ms-3x_coco': 0.05702888431833751, 'faster-rcnn_regnetx-800MF_fpn_ms-3x_coco': 0.03268460017531665, 'faster-rcnn_res2net-101_fpn_2x_coco': 0.08046177607863697, 'faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco': 0.08382054940978093, 'faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco': 0.06312783796395828, 'faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco': 0.09542835648380109, 'faster-rcnn_x101-32x4d_fpn_1x_coco': 0.079818383971257, 'faster-rcnn_x101-32x4d_fpn_2x_coco': 0.07870227899124373, 'faster-rcnn_x101-32x4d_fpn_ms-3x_coco': 0.07998388916698855, 'faster-rcnn_x101-32x8d_fpn_ms-3x_coco': 0.1156238762300406, 'faster-rcnn_x101-64x4d_fpn_1x_coco': 0.11734628321519538, 'faster-rcnn_x101-64x4d_fpn_2x_coco': 0.11433736957720857, 'faster-rcnn_x101-64x4d_fpn_ms-3x_coco': 0.11632367390305248}

yolo_in_total = {'yolov3_d53_320_273e_coco': 0.01166779247682486, 'yolov3_d53_mstrain-416_273e_coco': 0.00968256993080253, 'yolov3_d53_mstrain-608_273e_coco': 0.011311303323774195, 'yolov3_d53_fp16_mstrain-608_273e_coco': 0.01105828783405361, 'yolov3_mobilenetv2_8xb24-320-300e_coco': 0.00920141632877179, 'yolov3_mobilenetv2_8xb24-ms-416-300e_coco': 0.00911195242582862}

yolo_by_model_frame = {'yolov3_d53_320_273e_coco': [0.02610039710998535, 0.01116180419921875, 0.01000213623046875, 0.01490163803100586, 0.010034799575805664, 0.009727954864501953, 0.009858369827270508, 0.013640642166137695, 0.010326862335205078, 0.010055303573608398, 0.010289907455444336, 0.010625123977661133, 0.010465621948242188, 0.010602712631225586, 0.010573625564575195, 0.011847257614135742, 0.010636091232299805, 0.010245561599731445, 0.010095596313476562, 0.010090827941894531, 0.009957551956176758, 0.010035276412963867, 0.010093450546264648, 0.01002812385559082, 0.009949445724487305, 0.010158538818359375, 0.013103723526000977, 0.012962579727172852, 0.010308504104614258, 0.010132074356079102, 0.010164022445678711, 0.009549140930175781, 0.00952601432800293, 0.00966501235961914, 0.0098114013671875, 0.009508371353149414, 0.009463787078857422, 0.009590387344360352, 0.009556770324707031, 0.009526968002319336, 0.009506702423095703, 0.009454488754272461, 0.009445905685424805, 0.0094451904296875, 0.009453535079956055, 0.009421825408935547, 0.009429693222045898, 0.009447336196899414, 0.009393453598022461, 0.00940394401550293, 0.010592222213745117, 0.01110982894897461, 0.009329557418823242, 0.009345293045043945, 0.009370803833007812, 0.015501976013183594, 0.012247085571289062, 0.010131359100341797, 0.009731531143188477, 0.009572505950927734, 0.009528875350952148, 0.00949859619140625, 0.009443044662475586, 0.009410381317138672, 0.009389162063598633, 0.009531974792480469, 0.009377479553222656], 'yolov3_d53_mstrain-416_273e_coco': [0.01595473289489746, 0.009718894958496094, 0.009515523910522461, 0.009467363357543945, 0.008956193923950195, 0.008960962295532227, 0.010014772415161133, 0.008945703506469727, 0.009506702423095703, 0.009461402893066406, 0.009474039077758789, 0.009603738784790039, 0.009462594985961914, 0.009451150894165039, 0.009350299835205078, 0.008916854858398438, 0.015108585357666016, 0.01050567626953125, 0.009514331817626953, 0.00961756706237793, 0.00986337661743164, 0.010055780410766602, 0.009798049926757812, 0.009800195693969727, 0.009879589080810547, 0.011460304260253906, 0.009734392166137695, 0.009675741195678711, 0.009687423706054688, 0.00958871841430664, 0.009573221206665039, 0.009463787078857422, 0.00989842414855957, 0.009511232376098633, 0.009481191635131836, 0.009485721588134766, 0.009488582611083984, 0.009504318237304688, 0.00989389419555664, 0.009469270706176758, 0.009535074234008789, 0.010500907897949219, 0.00949859619140625, 0.009470701217651367, 0.009537696838378906, 0.009579658508300781, 0.009587764739990234, 0.009937047958374023, 0.009619474411010742, 0.009573698043823242, 0.009199142456054688, 0.009396791458129883, 0.010697126388549805, 0.008963584899902344, 0.008958578109741211, 0.009500503540039062, 0.009471893310546875, 0.009483814239501953, 0.009471893310546875, 0.009490013122558594, 0.009442806243896484, 0.009426116943359375, 0.009479999542236328, 0.00948953628540039, 0.009390115737915039, 0.009447813034057617, 0.009506940841674805], 'yolov3_d53_mstrain-608_273e_coco': [0.014862298965454102, 0.012098312377929688, 0.01090097427368164, 0.010930299758911133, 0.010946512222290039, 0.010581493377685547, 0.010610342025756836, 0.010589838027954102, 0.010589838027954102, 0.010611772537231445, 0.010563373565673828, 0.010707378387451172, 0.011674880981445312, 0.011066436767578125, 0.010890722274780273, 0.012283563613891602, 0.010839462280273438, 0.012066125869750977, 0.010753631591796875, 0.010704994201660156, 0.010661840438842773, 0.010606050491333008, 0.011668205261230469, 0.010806560516357422, 0.010550975799560547, 0.010601282119750977, 0.010637998580932617, 0.010586977005004883, 0.013003349304199219, 0.010610103607177734, 0.010532617568969727, 0.010511636734008789, 0.010625123977661133, 0.012044906616210938, 0.012246847152709961, 0.011687040328979492, 0.010845422744750977, 0.011142730712890625, 0.010910511016845703, 0.010928869247436523, 0.011050939559936523, 0.011059284210205078, 0.010659933090209961, 0.01157069206237793, 0.010846614837646484, 0.010619401931762695, 0.01091909408569336, 0.010884284973144531, 0.010924339294433594, 0.010728597640991211, 0.010636568069458008, 0.01065969467163086, 0.011330127716064453, 0.011500358581542969, 0.013018131256103516, 0.010721206665039062, 0.012036323547363281, 0.010685920715332031, 0.010535717010498047, 0.015109539031982422, 0.010514497756958008, 0.010890960693359375, 0.010589838027954102, 0.010862112045288086, 0.010512828826904297, 0.010670900344848633, 0.012495279312133789], 'yolov3_d53_fp16_mstrain-608_273e_coco': [0.014004945755004883, 0.011424779891967773, 0.010516166687011719, 0.010661840438842773, 0.010670900344848633, 0.01045536994934082, 0.010514020919799805, 0.01052403450012207, 0.010470151901245117, 0.010486364364624023, 0.011982202529907227, 0.010835409164428711, 0.011422872543334961, 0.011586904525756836, 0.011638641357421875, 0.01068735122680664, 0.010753393173217773, 0.010629892349243164, 0.010811805725097656, 0.01101994514465332, 0.010611295700073242, 0.01047658920288086, 0.01047205924987793, 0.010535955429077148, 0.01490020751953125, 0.015825271606445312, 0.010972261428833008, 0.010755777359008789, 0.010614871978759766, 0.010633468627929688, 0.011529922485351562, 0.013357877731323242, 0.011095046997070312, 0.01062774658203125, 0.010500669479370117, 0.010608434677124023, 0.010629892349243164, 0.010861873626708984, 0.01086282730102539, 0.010629415512084961, 0.01209402084350586, 0.010767459869384766, 0.010764122009277344, 0.010848522186279297, 0.011815309524536133, 0.010855436325073242, 0.01076650619506836, 0.012799263000488281, 0.011482954025268555, 0.010446548461914062, 0.010378360748291016, 0.012713909149169922, 0.010723114013671875, 0.010738849639892578, 0.01082468032836914, 0.010741233825683594, 0.010807514190673828, 0.01050114631652832, 0.011022567749023438, 0.010558843612670898, 0.010452985763549805, 0.010448932647705078, 0.010450124740600586, 0.010475635528564453, 0.010481595993041992, 0.010493040084838867, 0.010532855987548828], 'yolov3_mobilenetv2_8xb24-320-300e_coco': [0.013730287551879883, 0.009215354919433594, 0.009889841079711914, 0.009986639022827148, 0.012279272079467773, 0.009240865707397461, 0.009302616119384766, 0.034722089767456055, 0.009026527404785156, 0.009098529815673828, 0.00930929183959961, 0.012485027313232422, 0.009280681610107422, 0.00927734375, 0.009285926818847656, 0.009379148483276367, 0.009581565856933594, 0.009669303894042969, 0.011878490447998047, 0.008943557739257812, 0.009018182754516602, 0.009130477905273438, 0.00848388671875, 0.008502960205078125, 0.00908041000366211, 0.009215116500854492, 0.009102821350097656, 0.009091615676879883, 0.009087324142456055, 0.009084224700927734, 0.009067058563232422, 0.009061574935913086, 0.009051322937011719, 0.009126424789428711, 0.009157180786132812, 0.009376287460327148, 0.00918269157409668, 0.009610176086425781, 0.008615732192993164, 0.008705615997314453, 0.008629322052001953, 0.008651494979858398, 0.008557796478271484, 0.00861501693725586, 0.008821964263916016, 0.008565425872802734, 0.008632183074951172, 0.008665084838867188, 0.008496522903442383, 0.00854802131652832, 0.008460521697998047, 0.008636474609375, 0.008503437042236328, 0.009140729904174805, 0.010752677917480469, 0.00845026969909668, 0.008802413940429688, 0.010302066802978516, 0.008815765380859375, 0.008553504943847656, 0.008927583694458008, 0.008527994155883789, 0.008570671081542969, 0.008565187454223633, 0.008522272109985352, 0.008978843688964844, 0.009537696838378906], 'yolov3_mobilenetv2_8xb24-ms-416-300e_coco': [0.013756513595581055, 0.009609460830688477, 0.00955057144165039, 0.009577035903930664, 0.00950002670288086, 0.009483814239501953, 0.009423255920410156, 0.009456157684326172, 0.00943136215209961, 0.009378671646118164, 0.009372234344482422, 0.010108709335327148, 0.011142492294311523, 0.009714365005493164, 0.010976552963256836, 0.010469198226928711, 0.012392759323120117, 0.009816646575927734, 0.00897216796875, 0.008934736251831055, 0.01036381721496582, 0.009706258773803711, 0.00967860221862793, 0.009627103805541992, 0.009759664535522461, 0.009653806686401367, 0.009580612182617188, 0.009578704833984375, 0.009861230850219727, 0.009315013885498047, 0.009183883666992188, 0.009224414825439453, 0.009194612503051758, 0.009230613708496094, 0.009229183197021484, 0.009226560592651367, 0.009581565856933594, 0.009594202041625977, 0.00964808464050293, 0.009553194046020508, 0.009542465209960938, 0.009563446044921875, 0.009549617767333984, 0.00952768325805664, 0.009713411331176758, 0.009174823760986328, 0.009177446365356445, 0.00912022590637207, 0.009134769439697266, 0.009131669998168945, 0.009110450744628906, 0.0091094970703125, 0.009108781814575195, 0.009130001068115234, 0.009071588516235352, 0.009143352508544922, 0.009151935577392578, 0.010324239730834961, 0.009470701217651367, 0.009598493576049805, 0.009282588958740234, 0.009279012680053711, 0.009746551513671875, 0.009293556213378906, 0.009287595748901367, 0.009326457977294922, 0.009104013442993164]}
multi_take_time = np.array([0.08700728416442871, 0.0640566349029541, 0.06130552291870117, 0.06069207191467285, 0.06383085250854492, 0.058226823806762695, 0.060938119888305664, 0.0597081184387207, 0.05568361282348633, 0.0603480339050293, 0.06290698051452637, 0.0634005069732666, 0.057166099548339844, 0.06635212898254395, 0.057100772857666016, 0.06151008605957031, 0.05676579475402832, 0.05834174156188965, 0.058753252029418945, 0.05646085739135742, 0.05734539031982422, 0.05903434753417969, 0.05596566200256348, 0.06652116775512695, 0.06093454360961914, 0.05741548538208008, 0.059452056884765625, 0.055762290954589844, 0.05695152282714844, 0.05918574333190918, 0.059594154357910156, 0.05914640426635742, 0.05833888053894043, 0.058202505111694336, 0.05925393104553223, 0.05990409851074219, 0.058708906173706055, 0.05847477912902832, 0.05871176719665527, 0.05563640594482422, 0.06163907051086426, 0.057387590408325195, 0.056917667388916016, 0.05686163902282715, 0.055632829666137695, 0.056087493896484375, 0.05639147758483887, 0.061322927474975586, 0.06098365783691406, 0.05634260177612305, 0.06038689613342285, 0.05571126937866211, 0.06268572807312012, 0.057825326919555664, 0.05911827087402344, 0.059105873107910156, 0.05731534957885742, 0.05767035484313965, 0.05618119239807129, 0.05578112602233887, 0.058188676834106445, 0.05722808837890625, 0.05634951591491699, 0.06178164482116699, 0.059348344802856445, 0.056049346923828125, 0.05958700180053711])

yolo_by_frame = np.array([value for value in list(yolo_by_model_frame.values())])
yolo_by_frame = np.sum(yolo_by_frame, axis=0)
print(multi_take_time - yolo_by_frame)