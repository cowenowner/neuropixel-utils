function data = commonRegressionResidualsEachBank(imec, data, chIds, timeIdx, extraArg) %#ok<INUSL>
if nargin < 5
    extraArg = struct();
end
forbidden_radius = 5;
outer_radius = 20;
skip = 10000; % skip every skip points
lambda = 1e-03;
chanMaskGood = ismember(chIds, imec.goodChannels);
% This is laborious, but we do a regression, find the n channels say
% that are decent predictors for target channel. That becomes the
% model. The residuals are what is the 're-referenced' channel.
ch_ix = find(chanMaskGood);
data2 = data;
for iCh = 1:length(ch_ix)
    ch = chIds(ch_ix(iCh));
    permitted_ch_mask = chIds < ch-forbidden_radius | chIds > ch+forbidden_radius;
    bad_outer = chIds > (ch+outer_radius) | chIds < (ch-outer_radius);
    permitted_ch_mask = permitted_ch_mask & ~bad_outer;
    X = single(data(permitted_ch_mask,1:skip:end)');
    y = single(data(ch_ix(iCh),1:skip:end)');
    %         [b,bint,r,rint,stats] = regress(y,X);
    % warning off
    mod = fitlm(X,y); % get a lot of rank defficient errors. Not sure why.
    % warning on
    y_hat = predict(mod,data(permitted_ch_mask,:)');
    data2(ch_ix(iCh),:) = data(ch_ix(iCh),:) - int16(y_hat)';
    %         figure
    %         plot(data(ch_ix(iCh),1:10000))
    %         hold on
    %         plot(data2(ch_ix(iCh),1:10000))
    %         rms(data(ch_ix(iCh),1:10000))
    %         rms(data2(ch_ix(iCh),1:10000))
end
data = data2;
%     % subtract median of each channel over time
%     data(chanMaskGood, :) = bsxfun(@minus, data(chanMaskGood, :), median(data(chanMaskGood, :), 2));
%
%     % subtract median across good channels
%     data(chanMaskGood, :) = bsxfun(@minus, data(chanMaskGood, :), median(data(chanMaskGood, :), 1));
%
% then do Siegle et al. 2019 style median subtraction over simultaneously acquired samples
if contains(imec.channelMap.name, 'phase3a', 'IgnoreCase', true)
    for n = 1:24
        chIdsThisGroup = n:24:384;
        chanMaskThisGroup = ismember(chIds, chIdsThisGroup) & chanMaskGood;
        data(chanMaskThisGroup, :) = data(chanMaskThisGroup, :) - median(data(chanMaskThisGroup, :), 1, 'omitnan');
    end
end

% optionally do high pass filter on data as vanilla KS 2 would, this is useful to eliminate post-stim baseline drift
if isfield(extraArg, 'hp_filter') && extraArg.hp_filter
    [b, a] = butter(extraArg.hp_filter_half_order, extraArg.hp_filter_corner/extraArg.fs*2, 'high');
    data(chanMaskGood, :) = filter(b, a, data(chanMaskGood, :), [], 2); % causal forward filter
    data(chanMaskGood, :) = fliplr(filter(b, a, fliplr(data(chanMaskGood, :)), [], 2)); % acausal reverse filter
end
end

