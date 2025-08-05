function SSNR_value = compute_SSNR(cleanSpeech, processedSpeech, fs, segmentLength)
    % Compute Segmental SNR (SSNR)
    % cleanSpeech: Original clean speech signal
    % processedSpeech: Noise-reduced speech signal
    % fs: Sampling frequency
    % segmentLength: Length of each segment in milliseconds
    
    % Convert segment length from ms to samples
    segmentSamples = round((segmentLength / 1000) * fs);
    
    % Ensure signals are of equal length
    minLength = min(length(cleanSpeech), length(processedSpeech));
    cleanSpeech = cleanSpeech(1:minLength);
    processedSpeech = processedSpeech(1:minLength);
    
    % Initialize variables
    numSegments = floor(minLength / segmentSamples);
    SSNR_segments = zeros(1, numSegments);
    
    % Loop through each segment
    for k = 1:numSegments
        % Extract segment
        startIdx = (k - 1) * segmentSamples + 1;
        endIdx = startIdx + segmentSamples - 1;
        
        % Compute numerator (clean signal energy)
        numerator = sum(cleanSpeech(startIdx:endIdx).^2);
        
        % Compute denominator (difference energy)
        denominator = sum((cleanSpeech(startIdx:endIdx) - processedSpeech(startIdx:endIdx)).^2);
        
        % Avoid division by zero
        if denominator > 0
            SSNR_segments(k) = 10 * log10(numerator / denominator);
        else
            SSNR_segments(k) = 0; % Assign 0 dB if denominator is zero
        end
    end
    
    % Compute average SSNR
    SSNR_value = mean(SSNR_segments);
end
