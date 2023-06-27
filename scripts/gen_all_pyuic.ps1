$uis = Get-ChildItem -Recurse -Include *.ui
foreach ($in in $uis)
{
    $out = $in.DirectoryName + "\" + $in.BaseName + "_ui.py"
    Write-Host "Converting UI: $in"
    pyside6-uic "$in" -o "$out"
}
