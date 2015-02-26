#!/usr/bin/env python
"""Extracts and sums runtimes from a log file."""

import argparse
import re

class TimeRecord:
	"""Stores all information about a defined time tag, which should be extracted."""
	
	def __init__(self, pattern, name):
		self._pattern = re.compile(pattern)
		self._name = name
		self._times = []
		
	@property
	def pattern(self):
		return self._pattern
	
	@pattern.setter
	def pattern(self, value):
		self._pattern = re.compile(value)
		
	def match(self, line):
		m = self._pattern.match(line)
		time = -1
		if m is not None:
			time = float(m.group(1))
		return time
		
	@property
	def name(self):
		return self._name
	
	@name.setter
	def name(self, value):
		self._name = value

	def add_time(self, value):
		self._times.append(value)
		
	def get_sum(self):
		return sum(self._times) 
	
	def get_count(self):
		return len(self._times)
		
	def get_average(self):
		if(self.get_count() == 0):
			return 0
		return self.get_sum() / self.get_count()

	def get_median(self):
		if(self.get_count() == 0):
			return 0
		self._times.sort()
		return self._times[self.get_count() / 2]
		
	def remove_high_spikes(self, count=1):
		self._times.sort()
		while count > 0 and self.get_count() > 1:
			self._times.pop()
			count = count - 1
			
	def remove_low_spikes(self, count=1):
		self._times.sort()
		self._times.reverse()
		while count > 0 and self.get_count() > 1:
			self._times.pop()
			count = count - 1


def main():
	# CLI arguments
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('input', help='Input log file.')
	parser.add_argument('output', help='Output file. Will be overwritten!')
	args = parser.parse_args()
	
	# TODO place your regex here to extract the times
	records = []
	records.append(TimeRecord('.*time_foo: (.*) ms', 'foo(): '))
	records.append(TimeRecord('.*time_bar: (.*) ms', 'bar(): '))
	records.append(TimeRecord('.*time_baz: (.*) ms', 'baz(): '))
	
	# Open file
	file_in = open(args.input, 'r')
	file_out = open(args.output, 'w')
	
	# Parse file
	for line in file_in:
		for rec in records:
			time = rec.match(line)
			if 0 < time:
				rec.add_time(time)
	
	# Write results
	for rec in records:
		file_out.write('Count ' + rec.name + str(rec.get_count()) + '\n')
		file_out.write('Time ' + rec.name + str(rec.get_sum()) + '\n')
		file_out.write('Average ' + rec.name + str(rec.get_average()) + '\n')
		file_out.write('Median ' + rec.name + str(rec.get_median()) + '\n')
		file_out.write('\n')
	
	# Close file
	file_out.close()
	file_in.close()


if __name__ == "__main__":
	main()
