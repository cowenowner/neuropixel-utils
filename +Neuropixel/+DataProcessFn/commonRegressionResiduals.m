function data = commonRegressionResiduals(imec, data, chIdx, timeIdx) %#ok<INUSD>
% Cowen 2022
% TODO: We could speed things up considerably but developing the model on
% the first chunk alone and then applying to the remaining chunks.
% this could potentially be done using a persistent variable.
% Alternatively, we could first load in a subset of the ENTIRE dataset,
% develop the models for each channel, and then pass it along.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% persistent n_calls
% persistent mod
% if isempty(n_calls)
%     n_calls = 0;
mod = cell(length(chIdx),1);
% end
% n_calls = n_calls+1;
% % fprintf('%d,',n_calls);

chanMask = ismember(chIdx, imec.goodChannels);
% regression_type = 'linear';
regression_type = 'stepwise';

data(~chanMask,:) = 0; % turn the bad channels into zeros.
forbidden_radius = 5;
outer_radius = 23;
skip = 10000; % skip every skip points

ch_ix = find(chanMask);
data2 = data;
%
for iCh = 1:length(ch_ix)
    ch = chIdx(ch_ix(iCh));
    permitted_ch_mask = chIdx < ch-forbidden_radius | chIdx > ch+forbidden_radius;
    bad_outer = chIdx > (ch+outer_radius) | chIdx < (ch-outer_radius);
    permitted_ch_mask = permitted_ch_mask & ~bad_outer;
    warning off
    %     if n_calls== 1 || n_calls/20 == 0 % intermittently update the model. Not sure if this significantly speeds things up. The model really only takes about 2 seconds to compute per block.
    % Consequently, it might be best just to compute a new model for
    % each block OR just compute it once? Not sure.
    switch regression_type
        case 'linear'
            %,'RobustOpts','on' % does not seem to help at all.
            mod{iCh} = fitlm(single(data2(permitted_ch_mask,1:skip:end)'),single(data2(ch_ix(iCh),1:skip:end)')); % get a lot of rank defficient errors. Not sure why.
        case 'stepwise'
            mod{iCh} = stepwiselm(single(data2(permitted_ch_mask,1:skip:end)'),single(data2(ch_ix(iCh),1:skip:end)'),'Verbose',0);
        otherwise
            error('unknown regression type.')
    end
    %     end
    % stepwise might be OK. Might generalize better or not be as corrupted
    % by noisy bad channels as those channels will be deleted.
    warning on
    y_hat = predict(mod{iCh},data(permitted_ch_mask,:)');
    data(ch_ix(iCh),:) = data2(ch_ix(iCh),:) - int16(y_hat)';
end
end