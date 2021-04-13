def LZ77(input_text, l, look_ahead_length):

    sliding_window = l
    encoded = []
    searchBuffer = ""
    look_ahead_buffer = input_text[:look_ahead_length]
    input_text = input_text[look_ahead_length:]
    
    while len(look_ahead_buffer) > 0:

        tmpSubstring = look_ahead_buffer[:-1]
        while(len(tmpSubstring) > 1):

            # prefix := longest prefix of input that begins in window
            prefix = findLongestPrefix(searchBuffer, tmpSubstring)
            # if prefix exists in window then
            if(prefix != -1):
                # d := distance to the start of the prefix
                d = len(searchBuffer) - prefix
                # l := length of prefix
                l = len(prefix)
                
                break

            else:
                # c := first symbol of input
                tmpSubstring = tmpSubstring[0] 
                # d and l equal to 0 
                d = 0
                l = 0 
          
        # c := char following the prefix in input
        c = look_ahead_buffer[l]

        #append (d, l, c) to encoded input
        encoded.append((d, l, c))
        #shift the sliding window by l + 1 symbols 
        sliding_window = l + 1
        # discard l + 1 symbols from the beginning of window and add the l + 1 
        # first symbols of the input at the end of the window
        look_ahead_buffer = look_ahead_buffer[sliding_window:]
        look_ahead_buffer = look_ahead_buffer + input_text[:sliding_window]
        input_text = input_text[l + 1:]

    return encoded


def findLongestPrefix(text, char):
    index = 0
    if char in text:
        c = char[0]
        for ch in text:
            if ch == c:
                if text[index:index+len(char)] == char:
                    return index

            index = index + 1

    return -1


def LZ77_decoder(encoded):

    decoded = ""
    for code in encoded:
        d = code[0]
        l = code[1]
        if (d == 0):
            decoded = decoded + code[2]
        else:
            start = len(decoded) - d
            end = start + l
            added = decoded[start:end]
            added = added[:l]
            decoded = decoded + added + code[2]
    return decoded



