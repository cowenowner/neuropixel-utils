function data = commonAverageReferenceMovMean(imec, data, chIdx, timeIdx) %#ok<INUSD>
    chanMask = ismember(chIdx, imec.goodChannels);

    data(chanMask, :) = data(chanMask, :) - int16(movmedian(data(chanMask, :),15000, 2));
    % subtract median of each channel over time
    %     data(chanMask, :) = bsxfun(@minus, data(chanMask, :), median(data(chanMask, :), 2));
    % subtract median across good channels
    data(chanMask, :) = bsxfun(@minus, data(chanMask, :), median(data(chanMask, :), 1));
end

