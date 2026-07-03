#If WinActive("ahk_exe chrome.exe") || WinActive("ahk_exe firefox.exe")

^!s:: ; Ctrl + Alt + S
    ; --- Step 1: Get current URL ---
    Send, !d
    Sleep, 200
    Send, ^c
    ClipWait, 1
    url := Clipboard

    ; --- Step 2: Extract number and optional letter ---
    RegExMatch(url, "/(player|prospect)/(\d+)(?:/([HD]))?$", match)
    playerNum := match2
    letter := match3

    if (!playerNum) {
        MsgBox, Failed to parse URL: %url%
        return
    }

    saveFolder := "C:\Users\branh\Documents\Hardwood PROJECTSSSSSS\StatPredictor-Manual\PlayersInputted\"

    ; --- Step 2.5: If no letter, go to /H first ---
    if (!letter) {
        letter := "H"
        url := url "/H"
        Send, !d
        Sleep, 200
        Clipboard := url
        Sleep, 200
        Send, ^v
        Sleep, 100
        Send, {Enter}
        Sleep, 2500
    }

    ; --- Step 3: Save current page ---
    Send, ^s
    Sleep, 1200
    filename := saveFolder playerNum "-" letter ".html"
    Clipboard := filename
    Sleep, 200
    Send, ^v
    Sleep, 200
    Send, {Enter}
    Sleep, 1500

    ; --- Step 4: Go to other URL ---
    if (letter = "H")
        newLetter := "D"
    else
        newLetter := "H"

    newURL := RegExReplace(url, "[HD]$", newLetter)
    Send, !d
    Sleep, 200
    Clipboard := newURL
    Sleep, 200
    Send, ^v
    Sleep, 100
    Send, {Enter}

    ; --- Step 5: Save second page ---
    Sleep, 2500
    Send, ^s
    Sleep, 1200
    filename := saveFolder playerNum "-" newLetter ".html"
    Clipboard := filename
    Sleep, 200
    Send, ^v
    Sleep, 200
    Send, {Enter}
return
