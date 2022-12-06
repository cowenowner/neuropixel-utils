function data = commonRegressionResiduals(imec, data, chIdx, timeIdx) %#ok<INUSD>
% Cowen 2022
chanMask = ismember(chIdx, imec.goodChannels);
forbidden_radius = 5;
outer_radius = 16;
skip = 10000; % skip every skip points

ch_ix = find(chanMask);
data2 = data;
for iCh = 1:length(ch_ix)
    ch = chIdx(ch_ix(iCh));
    permitted_ch_mask = chIdx < ch-forbidden_radius | chIdx > ch+forbidden_radius;
    bad_outer = chIdx > (ch+outer_radius) | chIdx < (ch-outer_radius);
    permitted_ch_mask = permitted_ch_mask & ~bad_outer;
    warning off
    mod = fitlm(single(data2(permitted_ch_mask,1:skip:end)'),single(data2(ch_ix(iCh),1:skip:end)')); % get a lot of rank defficient errors. Not sure why.
    warning on
    y_hat = predict(mod,data(permitted_ch_mask,:)');
    data(ch_ix(iCh),:) = data2(ch_ix(iCh),:) - int16(y_hat)';
    %         figure
    %         plot(data(ch_ix(iCh),1:10000))
    %         hold on
    %         plot(data2(ch_ix(iCh),1:10000))
    %         rms(data(ch_ix(iCh),1:10000))
    %         rms(data2(ch_ix(iCh),1:10000))
end
end