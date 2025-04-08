%% 1. Load Data
genetic_data = readtable('Data/Genetics/16p11.2/Brain Canada - 16p11.2 demographics, IQ, dx.csv');

% Display the first few rows and variable names
head(genetic_data)
disp(genetic_data.Properties.VariableNames)

% Define the output directory
output_dir = 'Output/Brain_Canada_16p112_demog/all_BC/';

% Ensure the directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end


%% 2. Data Preprocessing

%% Age
% Define the current date
current_date = datetime('today');

% Convert the DateOfBirth column to datetime format
genetic_data.DateOfBirth = datetime(genetic_data.DateOfBirth, 'InputFormat', 'yyyy-MM-dd'); % Adjust format if needed

% Calculate age in years
genetic_data.Age = years(current_date - genetic_data.DateOfBirth);

% Round down to the nearest whole number
genetic_data.Age = floor(genetic_data.Age);

% Plot a histogram of the Age column
figure;
histogram(genetic_data.Age, 10); % Adjust the number of bins (10) as needed
title('Age Distribution');
xlabel('Age (years)');
ylabel('Frequency');
grid on;

% Save the histogram
saveas(gcf, fullfile(output_dir, 'age_histogram.png'));

%% Gender
% Count the occurrences of each gender
gender_counts = groupcounts(genetic_data.Gender); % Replace 'Gender' with the actual column name

% Get unique gender categories
gender_categories = unique(genetic_data.Gender);

% Plot a bar chart
figure;
bar(categorical(gender_categories), gender_counts);
title('Gender Distribution');
xlabel('Gender');
ylabel('Count');
grid on;

% Save the bar
saveas(gcf, fullfile(output_dir, 'gender_bar_chart.png'));

%% IQ
% Plot histogram for FSIQ
figure;
histogram(genetic_data.FSIQ, 10, 'FaceColor', 'b'); % Adjust bins as needed
title('Distribution of Full-Scale IQ (FSIQ)');
xlabel('FSIQ');
ylabel('Frequency');
grid on;

% Save the histogram
saveas(gcf, fullfile(output_dir, 'FSIQ_histogram.png'));

% Plot histogram for Non_VerbalIQ
figure;
histogram(genetic_data.Non_VerbalIQ, 10, 'FaceColor', 'g'); % Adjust bins as needed
title('Distribution of Non-Verbal IQ');
xlabel('Non-Verbal IQ');
ylabel('Frequency');
grid on;

% Save the histogram
saveas(gcf, fullfile(output_dir, 'Non_VerbalIQ_histogram.png'));

% Plot histogram for VerbalIQ
figure;
histogram(genetic_data.VerbalIQ, 10, 'FaceColor', 'r'); % Adjust bins as needed
title('Distribution of Verbal IQ');
xlabel('Verbal IQ');
ylabel('Frequency');
grid on;

% Save the histogram
saveas(gcf, fullfile(output_dir, 'VerbalIQ_histogram.png'));

%% Clinical diagnosis
columns_to_analyze = {
    'MotorDelay', ...
    'LanguageDelay', ...
    'HasTheParticipantReceivedAFormalDiagsisOfIntellectualDisability', ...
    'ASD', ...
    'ADD_ADHD', ...
    'LearningDisorder', ...
    'AnxietyDisorder', ...
    'SubstanceRelatedAndAddictiveDisorders', ...
    'OtherPsychiatricConditionOrComments'
};

% Loop through each column
for i = 1:length(columns_to_analyze)
    column_name = columns_to_analyze{i};
    
    % Filter rows where the column is not empty
    non_empty_rows = ~cellfun(@isempty, genetic_data.(column_name));
    empty_rows = cellfun(@isempty, genetic_data.(column_name));
    
    % Count the number of rows with and without a diagnosis
    num_with_diagnosis = sum(non_empty_rows);
    num_without_diagnosis = sum(empty_rows);
    total_rows = height(genetic_data);
    
    % Calculate proportions
    proportion_with_diagnosis = (num_with_diagnosis / total_rows) * 100;
    proportion_without_diagnosis = (num_without_diagnosis / total_rows) * 100;
    
    % Display results
    fprintf('Analysis for column: %s\n', column_name);
    fprintf('  With diagnosis: %d (%.2f%%)\n', num_with_diagnosis, proportion_with_diagnosis);
    fprintf('  Without diagnosis: %d (%.2f%%)\n', num_without_diagnosis, proportion_without_diagnosis);
    
    % Plot a pie chart for the proportions
    figure;
    pie([num_without_diagnosis, num_with_diagnosis], ...
        {sprintf('No Diagnosis (%d)', num_without_diagnosis), sprintf('Has Diagnosis (%d)', num_with_diagnosis)});
    title(['Proportion of Diagnoses for ', strrep(column_name, '_', ' ')]);

    % Save the pie chart
    saveas(gcf, fullfile(output_dir, [column_name, '_pie_chart.png']));
    
    
    % If there are non-empty rows, analyze the unique values
    if num_with_diagnosis > 0
        % Extract non-empty values
        filtered_data = genetic_data.(column_name)(non_empty_rows);
        
        % Count the occurrences of each unique value
        unique_values = unique(filtered_data); % Get unique values
        value_counts = cellfun(@(x) sum(strcmp(filtered_data, x)), unique_values); % Count occurrences
        
        % Calculate proportions for each unique value
        proportions = value_counts / num_with_diagnosis * 100; % Convert to percentages
        
        % Display detailed results for non-empty rows
        fprintf('  Breakdown of diagnoses:\n');
        for j = 1:length(unique_values)
            fprintf('    %s: %d (%.2f%%)\n', unique_values{j}, value_counts(j), proportions(j));
        end
        
        % Plot a bar chart for the absolute counts
        figure;
        bar(categorical(unique_values), value_counts);
        title(['Counts of Diagnoses for ', strrep(column_name, '_', ' ')]);
        xlabel('Category');
        ylabel('Count');
        grid on;

        % Save the bar chart
        saveas(gcf, fullfile(output_dir, [column_name, '_bar_chart.png']));
    end
end