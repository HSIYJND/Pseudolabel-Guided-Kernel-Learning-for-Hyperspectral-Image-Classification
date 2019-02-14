function [ Pred, F_pre] = ensemble_learning_add_meanck(no_classes,num_sampling,GroudTest,SVMResultTest_w_regu_v,SVMResultTest_s_regu_v,SVMResultTest_ws_regu_v,SVMResultTest2_w_v,SVMResultTest2_s_v,SVMResultTest_ws_v,SVMResultTest_w_mss_regu_v,SVMResultTest_wmss_v,SVMResultTest_mss_regu_v,SVMResultTest_mss_v)
%ENSEMBLE 此处显示有关此函数的摘要
%   此处显示详细说明

num_sample=size(GroudTest,1);
[pre_s_regu,f_pre_s_regu,oa_s_regu, aa_s_regu, K_s_regu, ua_s_regu]= uni_ensemble_learning(GroudTest,SVMResultTest_s_regu_v,num_sample, num_sampling,no_classes )
[pre_ws_regu,f_pre_ws_regu,oa_ws_regu, aa_ws_regu, K_ws_regu, ua_ws_regu]= uni_ensemble_learning(GroudTest,SVMResultTest_ws_regu_v,num_sample, num_sampling,no_classes )
[pre_w_regu,f_pre_w_regu,oa_w_regu, aa_w_regu, K_w_regu, ua_w_regu]= uni_ensemble_learning(GroudTest,SVMResultTest_w_regu_v,num_sample, num_sampling,no_classes )
[pre_w,f_pre_w,oa_w, aa_w, K_w, ua_w]= uni_ensemble_learning(GroudTest,SVMResultTest2_w_v,num_sample,num_sampling, no_classes )
[pre_s,f_pre_s,oa_s, aa_s, K_s, ua_s]= uni_ensemble_learning(GroudTest,SVMResultTest2_s_v,num_sample,num_sampling, no_classes )
[pre_ws,f_pre_ws,oa_ws, aa_ws, K_ws, ua_ws]= uni_ensemble_learning(GroudTest,SVMResultTest_ws_v,num_sample, num_sampling, no_classes )
[pre_wms_regu,f_pre_wms_regu,oa_wms_regu, aa_wms_regu, K_wms_regu, ua_wms_regu]=uni_ensemble_learning(GroudTest,SVMResultTest_w_mss_regu_v,num_sample, num_sampling, no_classes );
[pre_wms,f_pre_wms,oa_wms, aa_wms, K_wms, ua_wms]=uni_ensemble_learning(GroudTest,SVMResultTest_wmss_v,num_sample, num_sampling, no_classes );

[pre_ms,f_pre_ms,oa_ms, aa_ms, K_ms, ua_ms]=uni_ensemble_learning(GroudTest,SVMResultTest_mss_v,num_sample, num_sampling, no_classes );
[pre_ms_regu,f_pre_ms_regu,oa_ms_regu, aa_ms_regu, K_ms_regu, ua_ms_regu]=uni_ensemble_learning(GroudTest,SVMResultTest_mss_regu_v,num_sample, num_sampling, no_classes );

Pred.s_regu=pre_s_regu;
Pred.ws_regu=pre_ws_regu;
Pred.w_regu=pre_w_regu;
Pred.w=pre_w;
Pred.s=pre_s;
Pred.ws=pre_ws;
Pred.wms=pre_wms;
Pred.wms_regu=pre_wms_regu;
Pred.ms=pre_ms;
Pred.ms_regu=pre_ms_regu;

F_pre.s_regu=f_pre_s_regu;
F_pre.ws_regu=f_pre_ws_regu;
F_pre.w_regu=f_pre_w_regu;
F_pre.w=f_pre_w;
F_pre.s=f_pre_s;
F_pre.ws=f_pre_ws;

F_pre.wms=f_pre_wms;
F_pre.wms_regu=f_pre_wms_regu;
F_pre.ms=f_pre_ms;
F_pre.ms_regu=f_pre_ms_regu;






end

