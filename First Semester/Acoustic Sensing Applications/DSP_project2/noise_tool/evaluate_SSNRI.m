function SSNRI_value = evaluate_SSNRI(cleanSpeech, noisySpeech, processedSpeech, fs, segmentLength)
    % Evaluate Segmental Signal-to-Noise Ratio Improvement (SSNRI)
    % cleanSpeech: Clean speech signal
    % noisySpeech: Noisy speech signal
    % processedSpeech: Noise-reduced speech signal
    % fs: Sampling frequency
    % segmentLength: Length of each segment in milliseconds

    % Part 1: Compute SSNR of the noisy speech
    SSNR_noisy = compute_SSNR(cleanSpeech, noisySpeech, fs, segmentLength);

    % Part 2: Compute SSNR of the processed speech (after noise reduction)
    SSNR_processed = compute_SSNR(cleanSpeech, processedSpeech, fs, segmentLength);

    % Calculate SSNRI
    SSNRI_value = SSNR_processed - SSNR_noisy;

    % Display results
    disp(['SSNR of noisy speech: ', num2str(SSNR_noisy), ' dB']);
    disp(['SSNR of processed speech: ', num2str(SSNR_processed), ' dB']);
    disp(['Segmental SNR Improvement (SSNRI): ', num2str(SSNRI_value), ' dB']);
end

function SSNR_value = compute_SSNR(cleanSpeech, testSpeech, fs, segmentLength)
    % Compute Segmental SNR (SSNR)
    % cleanSpeech: Clean speech signal
    % testSpeech: Either noisy or processed speech
    % fs: Sampling frequency
    % segmentLength: Length of each segment in milliseconds

    % Convert segment length from ms to samples
    segmentSamples = round((segmentLength / 1000) * fs);

    % Ensure signals are of equal length
    minLength = min(length(cleanSpeech), length(testSpeech));
    cleanSpeech = cleanSpeech(1:minLength);
    testSpeech = testSpeech(1:minLength);

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
        denominator = sum((cleanSpeech(startIdx:endIdx) - testSpeech(startIdx:endIdx)).^2);

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
