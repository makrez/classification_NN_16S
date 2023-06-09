README for SILVA export files

RAST FILES:
===========

Specific export files for the MG-RAST server (Argonne National Lab)

TAXONOMY FILES:
===============

tax_slv_[ls]su_VERSION.txt
-------------------------
These files contain taxonomic rank designations for all taxonomic paths
used in the SILVA taxonomies. Additionally, a unique numeric identifier is 
assigned to each taxon (path). These identifiers will be mostly stable in 
upcoming SILVA releases. 

IDs used in the SSU and LSU files do not match.

Field description:
path:
    The full taxonomic path including the name of the group itself.
    Segments are separated with ";"
taxid:
    numerical identifier
rank:
    The rank designation. 
remark:
    Can be empty ('') or a or w.
    a: Marks taxa of environmental origin. That is, taxa containing no 
       sequence coming from a cultivated organism.
    w: Marks taxa scheduled for revision in the next release.
release:
    The SILVA release version

tax_slv_[ls]su_VERSION.diff
---------------------------
Difference between the current version of the SILVA taxonomy and the previous
release.

Field description:
status:
    the status of the taxono (+ added, - removed)
taxid:
    numerical identifier
path:
    the full path of the added/removed taxon

taxmap_TAXNAME_[ls]su_VERSION.txt
----------------------------
mapping of each entry in the SILVA database to a taxonomic path. Different
rRNA regions of the same INSDC entry (genome) may be assigned to multiple
paths (contaminations or micro diversity among the rRNA sequences).

The taxmap_embl* files contain the taxonomic path assigned by the original
submitter of the sequence. The last column in this file contains the numerical
ID of the NCBI taxonomy project assigned to this entry. This ID is extracted
from the source feature of an EMBL entry during the import of the sequence.

Field description:
pacc:
    INSDC primary accession
start
    start position of the rRNA region within the INSDC sequence entry
stop:
    stop position of the rRNA region within the INSDC sequence entry
path:
    taxonomic path assigned to the region
name
    the organism name assigned to the sequence
taxid:
    optional field containing the numerical ID of the taxonomic path


tax_TAXNAME_[ls]su_VERSION.{map,tre}
----------------------------
SILVA taxonomy in the Newick tree format and the corespoding numerical id to
taxonomic path mapping file (MEGAN compatible).

tax_TAXNAME_[ls]su_VERSION.acc_taxid
----------------------------
Mapping of 'SILVA' sequence IDs (<INSDC primary accession>.<start>.<stop>)
used in FASTA files to the numeric SILVA taxid (MEGAN compatible).




SEQUENCE FILES:
===============

*_tax_silva.fasta.gz
-----------------
Multi FASTA files of the SSU/LSU databases including the SILVA taxonomy for
Bacteria, Archaea and Eukaryotes in the header.

REMARK: The sequences in the files are NOT truncated to the effective LSU or
SSU genes. They contain the full entries as they have been deposited in the
public repositories (ENA/GenBank/DDBJ). 

Fasta header:
>accession_number.start_position.stop_position taxonomic path organism name

*_tax_silva_full_align_trunc.fasta.gz
-----------------------
Multi FASTA files of the SSU/LSU databases including the SILVA taxonomy for
Bacteria, Archaea and Eukaryotes in the header (including the FULL alignment).

REMARK: Sequences in these files haven been truncated. This means that all
nucleotides that have not been aligned were removed from the sequence.

*_tax_silva_trunc.fasta.gz
-----------------------
Multi FASTA files of the SSU/LSU database including the SILVA taxonomy for 
Bacteria, Archaea and Eukaryotes in the header.

REMARK: Sequences in these files haven been truncated. This means that all
nucleotides that have not been aligned were removed from the sequence.



CUSTOMISED FILES:
=================

*.acs
-----
Lists with all accession numbers in LSUParc and SSUParc

*.clstr
-------
Mapping of 'ref' sequences to 'nr' sequences. The file uses the CD-Hit file
format.

*quality*.csv
-------------
complete quality values for all SILVA Parc sequences Datasets.
Header: Primary Accession,Start,Stop,Region Length,Annotation Source,Sequence Quality,
% Ambiguities,% Homopolymers,% Vector Contamination,Alignment Quality, Base Pair Score,
# Aligned Bases,Pintail Quality



Directory 'User'
User specific exports done on request

Abbreviations:

LSU: Large subunit (23S/28S ribosomal RNAs)
SSU: Small subunit (16S/18S ribosomal RNAs)




Questions: contact@arb-silva.de
