import heapq
import codecs

def count_frequency(data):
        frequency = {}
        for char in text:
            if not char in frequency:
                frequency[char] = 0
            frequency[char] += 1
        return frequency


""" Implement encoding and compression"""
def Huffman_algorithm(input_path, output_path):

    """Read input data"""
    file = open(input_path, "r")
    data = file.read()


    """Take frequency of the characters and order in a heap"""
    dictionary = count_frequency(data)
    dictionary

    heap = [(value, key) for key,value in dictionary.items()]
    heapq.heapify(heap)

    """Generate encoding"""
    dictionary = dict()
    while len(heap)>1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        for i in left[1]:
            # Add 0 if encoding to the left
            dictionary[i] = '0' + dictionary.get(i,"")
            print(left)
        for i in right[1]:
            # Add 1 if encoding the right
            dictionary[i]= '1' + dictionary.get(i,"")
            print(right)
        heapq.heappush(heap,(left[0]+right[0], left[1]+right[1]))
    print(heap)
    print(dictionary)

    encoded = ""
    for char in data:
        encoded+=dictionary[char]
    encoded

    """Compress into an output file"""
    byte = 8 - len(encoded)%8
    for i in range(byte):
        encoded += "0"
	
    tmp = "{0:08b}".format(byte)
    encoded_text = tmp + encoded
    encoded_text

    if(len(encoded_text) % 8 != 0):
            print("ERROR")
            exit(0)

    compressed = bytearray()
    for i in range(0, len(encoded_text), 8):
        byte = encoded_text[i:i+8]
        compressed.append(int(byte, 2))
       
    with codecs.open(output_path, 'wb') as output:
        output.write(bytes(compressed))


"""Implement decompression"""
def Huffman_decoder (input_path, output_path):

    with open(input_path, 'rb') as file, codecs.open(output_path, 'w', encoding='latin-1') as output:
            
        bit_string = ""
        byte = file.read(1)
            
        while(len(byte) > 0):
            byte = ord(byte)
            bits = bin(byte)[2:].rjust(8, '0')
            bit_string += bits
            byte = file.read(1)
        
            padded_info = padded_encoded_text[:8]
            extra_padding = int(padded_info, 2)

            padded_encoded_text = padded_encoded_text[8:]
            encoded_text = padded_encoded_text[:-1*extra_padding]
    
    return encoded_text

"""Remarks: I tried with strings, its working. I ran it in colab 
   so I added files withput writing paths thats why here is not 
   the examples paths. Even if in the q1 is not asked to do compression 
   and decompression, they are here. Decoding is missing.