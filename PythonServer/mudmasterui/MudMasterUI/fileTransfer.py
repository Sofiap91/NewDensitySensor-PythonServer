import os
import time
import threading
from datetime import datetime

class File_Transfer(object):
    """
    *****************************************************************************
    @file   file_transfer.py
    @brief  Handles the transfer of new files from a source folder to a destination folder
            and removes files not from the current day in the source folder.
    *****************************************************************************
    Methods:
    __init__(self, app):
        Initializes the File_Transfer object with application context and starts
        a background thread to handle file transfer operations.

    copy_new_files(self):
        Checks the source folder for files from the current day and combines them into
        a single file in the destination folder.

    clean_old_files(self):
        Removes files that are not from the current day in the source folder.

    run(self):
        Runs in a separate thread to periodically check for and transfer new files
        and clean up old files from the source folder at regular intervals.
    """
    
    def __init__(self, app):
        """Initialize the File_Transfer object and start the file transfer thread."""
        
        self._app = app
        self.src_folder = self._app.config['MACHINE_DIRECTORY']
        self.dest_folder = self._app.config['TRANSFER_DIRECTORY']
        
        # Create and start a thread to run the file transfer process
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def copy_new_files(self):
        """Append each file from the source folder to a daily file in the destination folder and delete the file from the source folder."""
        
        def process_directory(src_dir, dest_dir):
            """Process each file and subdirectory in the source directory."""
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Get today's date for filename
            today_date = datetime.now().strftime('%Y-%m-%d')
            daily_file = os.path.join(dest_dir, f'transfer_{today_date}.csv')

            # Process files from the source directory
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)

                if os.path.isdir(src_item):
                    # Process subdirectories (you can add logic to exclude folders like 'calData' here)
                    process_directory(src_item, dest_dir)
                elif os.path.isfile(src_item) and item.endswith('.csv'):
                    try:
                        # Check if the file is from today (by its modification date)
                        file_date = datetime.fromtimestamp(os.path.getmtime(src_item)).strftime('%Y-%m-%d')

                        if file_date == today_date:
                            # Open the source file for reading
                            with open(src_item, 'r') as src_file:
                                lines = src_file.readlines()

                            # Check if the daily file exists
                            file_exists = os.path.exists(daily_file)
                            
                            # If the daily file does not exist, create it and write the header
                            if not file_exists:
                                with open(daily_file, 'w') as daily_file_create:
                                    daily_file_create.write(lines[0])  # Write the header from the first source file
                                start_index = 1  # Start appending from the second line (skip the header)
                            else:
                                # If the daily file exists, check if the header matches
                                with open(daily_file, 'r') as daily_file_read:
                                    daily_header = daily_file_read.readline().strip().split(',')
                                
                                # 
                                header = lines[0].strip().split(',')  # Assuming CSV with comma delimiter
                                if daily_header == header:
                                    start_index = 1  # Skip the header in the source file
                                else:
                                    # If the headers don't match, skip
                                    start_index = 0  # Start appending from the second line (skip the header)

                            # Open the daily file in append mode and append data without the header
                            with open(daily_file, 'a') as daily_file_append:
                                for line in lines[start_index:]:  # Skip the header of the source file
                                    daily_file_append.write(line)

                            print(f"Appended data from {item} to daily file")

                            # Delete the source file after appending the data
                            os.remove(src_item)
                            print(f"Deleted source file: {item}")

                    except Exception as e:
                        print(f"Error processing file {item}: {e}")
                        # Skip the file and continue to the next one
                        continue

        # Start processing from the root source and destination folders
        process_directory(self.src_folder, self.dest_folder)
        
    
    def clean_old_files(self):
        """Remove files that are not from the current day in the source folder."""
        
        today_date = datetime.now().strftime('%Y-%m-%d')
        
        for item in os.listdir(self.src_folder):
            src_item = os.path.join(self.src_folder, item)
            
            # If it's a file, check its modification date
            if os.path.isfile(src_item):
                file_date = datetime.fromtimestamp(os.path.getmtime(src_item)).strftime('%Y-%m-%d')
                
                # If the file is not from today, delete it
                if file_date != today_date:
                    try:
                        os.remove(src_item)
                        print(f"Deleted old file: {item}")
                    except Exception as e:
                        print(f"Error deleting file {item}: {e}")
                        # Skip this file and continue to the next one
                        continue

    def run(self):
        """Continuously check for new files and copy them at regular intervals."""
        while True:
            self.copy_new_files()  # Copy files and combine them into one for the current day
            self.clean_old_files()  # Clean up old files that are not from today
            print("Waiting for next cycle...")
            time.sleep(self._app.config['TRANSFER_DELAY'] * 60)  # Wait for the configured delay before the next cycle