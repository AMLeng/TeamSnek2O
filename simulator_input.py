import os
import shutil

# MAKE SURE TO INCLUDE THE SLASH AFTER THE FOLDER. Otherwise things get uhappy
IN_FOLDER = '/home/Desktop/drivindata/'
OUT_FOLDER = 'SimulatorData/'
CAMERAS = ['left', 'center', 'right']

def main(remove_old_data, verbose):
    # Removes old data in OUT_FOLDER if asked to. Very polite like that
    if remove_old_data:
        print('Deleting old data...')
        shutil.rmtree(OUT_FOLDER)

    # Set up the left, center, and right data directories
    for prefix in CAMERAS:
        if not os.path.exists(OUT_FOLDER + prefix + 'Images'):
            os.makedirs(OUT_FOLDER + prefix + 'Images')

    print('Copying images')
    # Copy the images into the right data directories
    for image_file in os.listdir(IN_FOLDER + 'IMG/'):
        # Assumes the image fits the format [camera]_[year]_[month]_[day]_[hour]_[minute]_[second]_[microseconds].jpg
        camera_angle = image_file.split('_')[0]
        if verbose:
            print('Copying ' + image_file + ' to ' + OUT_FOLDER + camera_angle + 'Images/' + image_file[camera_angle.__len__() + 1:])
            shutil.copyfile(IN_FOLDER + 'IMG/' + image_file, OUT_FOLDER + camera_angle + 'Images/' + image_file[camera_angle.__len__() + 1:])
        # Remove the camera from the start of the name
        # os.rename(OUT_FOLDER + camera_angle + 'Images/' + image_file, image_file[camera_angle.__len__() + 1:])

    # Open the driving log made by the simulator
    driving_log = open(IN_FOLDER + 'driving_log.csv', 'r')

    # For each camera, open a new log file
    camera_logs = []
    for camera in CAMERAS:
        camera_logs.append(open(OUT_FOLDER + camera + 'Images/approximatedStamps.txt', 'w'))

    print('Writing log files...')
    # Assumes that the first n columns in the log file are pathnames
    for line in driving_log:
        columns = line.split(',')
        for path in columns[0:CAMERAS.__len__()]:
            path = path[IN_FOLDER.__len__():]
            camera = path.split('_')[0]
            path = path[camera.__len__()+1:]

        degrees_to_turn = columns[CAMERAS.__len__()]

        for log in camera_logs:
            log.write(path + ' ' + degrees_to_turn + '\n')


    driving_log.close()
    for log in camera_logs:
        log.close()


main(True, True)