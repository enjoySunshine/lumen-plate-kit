CC ?= cc
CFLAGS ?= -O3 -DNDEBUG -std=gnu11 -fPIC -fvisibility=hidden -Wno-unused-parameter -Wall
CPPFLAGS += -Iinclude -Ithird_party/libyuv/include
LDLIBS += -lm -lz

SRCS = \
	src/alpha.c \
	src/avif.c \
	src/colr.c \
	src/colrconvert.c \
	src/diag.c \
	src/exif.c \
	src/gainmap.c \
	src/io.c \
	src/mem.c \
	src/obu.c \
	src/properties.c \
	src/rawdata.c \
	src/read.c \
	src/reformat.c \
	src/reformat_libsharpyuv.c \
	src/reformat_libyuv.c \
	src/sampletransform.c \
	src/scale.c \
	src/stream.c \
	src/utils.c \
	src/write.c \
	third_party/libyuv/source/scale.c \
	third_party/libyuv/source/scale_common.c \
	third_party/libyuv/source/scale_any.c \
	third_party/libyuv/source/row_common.c \
	third_party/libyuv/source/planar_functions.c

OBJS = $(SRCS:.c=.o)

TOOL = tools/lumen_plate_kit
TOOL_OBJ = tools/lumen_plate_kit.o

.PHONY: all clean view
all: libavif.a $(TOOL)

libavif.a: $(OBJS)
	ar rcs $@ $(OBJS)

$(TOOL): $(TOOL_OBJ) libavif.a
	$(CC) $(CFLAGS) -o $@ $(TOOL_OBJ) libavif.a $(LDLIBS)

tools/%.o: tools/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Regenerates every asset and timing shown in view/index.html from the plates.
view: $(TOOL)
	./$(TOOL)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) libavif.a $(TOOL) $(TOOL_OBJ)
