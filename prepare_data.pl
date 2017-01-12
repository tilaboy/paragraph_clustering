#!/usr/bin/env perl

use strict;
use warnings;

use lib "$ENV{TKHOME}/lib";
use utf8;
use MultiXMLFile;
use TK::FileSlurp qw/write_file/;

binmode STDERR, ":utf8";
binmode STDOUT, ":utf8";

my $input_mxml = shift;

my @sections =
  qw/personalsec coverlettersec educationsec experiencesec skillsec summambitsec extracurricularsec referencesec publicationsec/;

my $mxmlObj = new MultiXMLFile(
                                $input_mxml,
                                {
                                   topLevelTag => 'begin',
                                   fileID      => undef,
                                   attrID      => 'id'
                                }
);

my $index = 0;

while ( my $partXML = $mxmlObj->getNextPart() ) {
    $index++;

    # generate doc name
    for my $tagName (@sections) {
        my @found_sections = $partXML->select("//$tagName");
        for my $found_section (@found_sections) {
            my $section_text = $found_section->get_text();
            $section_text = lc($section_text);
            $section_text =~ s/\n/ /g;
            $section_text =~ s/\s+/ /xg;
            $section_text =~ s/['"]/'/g;
            $section_text =~ s/^\s+//xg;
            $section_text =~ s/\s+$//xg;

            print "$tagName\t$section_text" . "\n";
        }
    }
}
