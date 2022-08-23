from Code.functions.convert_annotations import convert_xml_annotation_to_mask

# define parameter
windowsize_r = 200
windowsize_c = 200
path_pictures = '../data/inter_annotator_agreement/unlabeled'
path_labeled = '../data/inter_annotator_agreement/labeled'

path_xml_file = '../data/inter_annotator_agreement/Export_roboflow/Teilbild_Oldenburg_00000006_00000011_1200_2200_Felix.xml'

convert_xml_annotation_to_mask(xml_file=path_xml_file,
                               path_picture=path_pictures,
                               path_export=path_labeled,
                               windowsize_r=windowsize_r,
                               windowsize_c=windowsize_c)


