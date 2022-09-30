#!/usr/bin/env bash

if [ "$USER" = "reinders" ]
then
    LUIS_USER=nhmlrein
else
    LUIS_USER=nhmlschf
fi
PROJECT_PATH=/bigwork/"$LUIS_USER"/projects/timm-vit
ssh luis -t -- "mkdir -p $PROJECT_PATH"
echo "copying code..."
rsync -vhra ./ luist:$PROJECT_PATH --include='**.gitignore' --exclude='/.git' --filter=':- .gitignore' --delete-after

if [ -n "$1" ]
then
echo "running command: $1"
ssh luis -t -- "bash -lc 'module load Miniconda3; cd $PROJECT_PATH || exit; conda activate timm && $1 && wandb sync'"
fi
echo "done"