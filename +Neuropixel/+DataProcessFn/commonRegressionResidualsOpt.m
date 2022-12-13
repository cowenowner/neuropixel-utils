function data = commonRegressionResiduals(imec, data, chIdx, timeIdx) %#ok<INUSD>
% Cowen 2022
% TODO: We could speed things up considerably but developing the model on
% the first chunk alone and then applying to the remaining chunks.
% this could potentially be done using a persistent variable.
% Alternatively, we could first load in a subset of the ENTIRE dataset,
% develop the models for each channel, and then pass it along.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent n_calls
persistent mod
if isempty(n_calls)
    n_calls = 0;
    mod = cell(length(chIdx),1);
end
n_calls = n_calls+1;
% fprintf('%d,',n_calls);

chanMask = ismember(chIdx, imec.goodChannels);
regression_type = 'linear';
% regression_type = 'stepwise';

data(~chanMask,:) = 0; % turn the bad channels into zeros.
forbidden_radius = 5;
outer_radius = 23;
skip = 2000; % skip every skip points

good_ch_ix = find(chanMask);
y_hat =  zeros(size(data),'int16'); % the prediction
%
for iCh = 1:length(good_ch_ix)
    row_ix = good_ch_ix(iCh);
    permitted_ch_mask = chIdx < (row_ix-forbidden_radius) | chIdx > (row_ix+forbidden_radius);
    bad_outer = chIdx > (row_ix+outer_radius) | chIdx < (row_ix-outer_radius);
    permitted_ch_mask = permitted_ch_mask & ~bad_outer;
    if n_calls== 1 % || n_calls/20 == 0 % intermittently update the model. Not sure if this significantly speeds things up. The model really only takes about 2 seconds to compute per block.
        warning off
        % Consequently, it might be best just to compute a new model for
        % each block OR just compute it once? Not sure.
        switch regression_type
            case 'linear_fast'
                % Maybe this is faster
                B = single(data(permitted_ch_mask,1:skip:end)')\single(data(row_ix,1:skip:end)');
            case 'linear'
                %,'RobustOpts','on' % does not seem to help at all.
                mod{row_ix} = fitlm(single(data(permitted_ch_mask,1:skip:end)'),single(data(row_ix,1:skip:end)')); % get a lot of rank defficient errors. Not sure why.
            case 'stepwise'
                mod{row_ix} = stepwiselm(single(data(permitted_ch_mask,1:skip:end)'),single(data(row_ix,1:skip:end)'),'Verbose',0);
            otherwise
                error('unknown regression type.')
        end
        warning on
    end
    % stepwise might be OK. Might generalize better or not be as corrupted
    % by noisy bad channels as those channels will be deleted.
    %     y_hat(row_ix,:) = int16(predict(mod{row_ix},data(permitted_ch_mask,:)'))';
    % feval is a tiny bit faster I believe than predict
  
%      y_hat(row_ix,:) = int16(data(permitted_ch_mask,:)'*B)';
     y_hat(row_ix,:) = int16(feval(mod{row_ix},data(permitted_ch_mask,:)'))';

end
data = data - y_hat;

end