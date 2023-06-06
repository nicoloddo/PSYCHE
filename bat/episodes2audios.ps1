# Check if ffmpeg is installed
if (!(Get-Command "ffmpeg" -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg could not be found, please install it first."
    Exit
}

# Get the list of .mkv files
$files = Get-ChildItem -Recurse -Filter *.mkv

# Loop through the files
foreach ($file in $files) {
    # Get the file name
    $filename = $file.Name

    # Extract the episode name
    if ($filename -match 'S\d+E\d+') {
        $episode = $Matches[0]
    } else {
        Write-Host "No episode number found in $filename"
        continue
    }

    # Convert the file to audio, downmixing to stereo
    $outputPath = Join-Path -Path $file.Directory.FullName -ChildPath "$episode.mp3"
    ffmpeg -i $file.FullName -vn -ac 2 -ar 22050 -y $outputPath
}
